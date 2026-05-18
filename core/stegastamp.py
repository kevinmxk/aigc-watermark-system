"""
StegaStamp 用户指纹模块
基于 TensorFlow SavedModel 的像素域水印编码与解码
"""
import os
import numpy as np
from PIL import Image
from typing import Optional, Dict, Tuple


class StegaStampEncoder:
    """StegaStamp 编码器：将用户 ID 比特串嵌入图像像素"""

    def __init__(self, encoder_path: str, height: int = 400, width: int = 400, secret_size: int = 100):
        """
        Args:
            encoder_path: Encoder SavedModel 路径
            height: 输入图像高度
            width: 输入图像宽度
            secret_size: 用户 ID 比特数
        """
        self.encoder_path = encoder_path
        self.height = height
        self.width = width
        self.secret_size = secret_size
        self._session = None
        self._graph = None
        self._enc_secret = None
        self._enc_image = None
        self._enc_out = None

    def _load_model(self):
        """延迟加载 TensorFlow 模型"""
        if self._session is not None:
            return

        # 必须在导入 TF 之前设置环境变量
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import warnings
        warnings.filterwarnings('ignore')

        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()
        from tensorflow.python.saved_model import tag_constants

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._session = tf.Session()
            tf.saved_model.loader.load(
                self._session,
                [tag_constants.SERVING],
                self.encoder_path
            )
            enc_graph = tf.get_default_graph()
            self._enc_secret = enc_graph.get_tensor_by_name('secret:0')
            self._enc_image = enc_graph.get_tensor_by_name('image:0')
            self._enc_out = enc_graph.get_tensor_by_name('clip_by_value:0')

    def encode(self, image: Image.Image, secret: np.ndarray) -> Tuple[Image.Image, np.ndarray]:
        """
        将用户 ID 比特串编码到图像中

        Args:
            image: 原始图像（PIL Image）
            secret: 用户 ID 比特串 (长度为 secret_size 的 numpy 数组)

        Returns:
            (编码后的图像, 编码后的 numpy 数组 [0,1])
        """
        self._load_model()

        # 调整图像尺寸
        img_resized = image.resize((self.width, self.height), Image.LANCZOS)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_arr, 0)

        # 编码
        with self._graph.as_default():
            enc_output = self._session.run(
                self._enc_out,
                feed_dict={
                    self._enc_secret: np.expand_dims(secret, 0),
                    self._enc_image: img_batch,
                }
            )

        enc_clipped = np.clip(enc_output[0], 0, 1)
        enc_image = Image.fromarray((enc_clipped * 255).astype(np.uint8))

        return enc_image, enc_clipped

    def close(self):
        """关闭 TensorFlow 会话"""
        if self._session is not None:
            self._session.close()
            self._session = None


class StegaStampDecoder:
    """StegaStamp 解码器：从图像中提取用户 ID 比特串"""

    def __init__(self, decoder_path: str, height: int = 400, width: int = 400, secret_size: int = 100):
        """
        Args:
            decoder_path: Decoder SavedModel 路径
            height: 输入图像高度
            width: 输入图像宽度
            secret_size: 用户 ID 比特数
        """
        self.decoder_path = decoder_path
        self.height = height
        self.width = width
        self.secret_size = secret_size
        self._session = None
        self._graph = None
        self._dec_image = None
        self._dec_out = None

    def _load_model(self):
        """延迟加载 TensorFlow 模型"""
        if self._session is not None:
            return

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import warnings
        warnings.filterwarnings('ignore')

        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()
        from tensorflow.python.saved_model import tag_constants

        self._graph = tf.Graph()
        with self._graph.as_default():
            self._session = tf.Session()
            tf.saved_model.loader.load(
                self._session,
                [tag_constants.SERVING],
                self.decoder_path
            )
            dec_graph = tf.get_default_graph()
            self._dec_image = dec_graph.get_tensor_by_name('image_dec:0')
            # 找到 Round 操作获取解码输出
            round_ops = [op for op in dec_graph.get_operations() if op.type == 'Round']
            self._dec_out = round_ops[-1].outputs[0]

    def decode(self, image: Image.Image) -> np.ndarray:
        """
        从图像中解码用户 ID 比特串

        Args:
            image: 编码后的图像（PIL Image）

        Returns:
            解码后的比特串 (长度为 secret_size 的 numpy 数组)
        """
        self._load_model()

        # 调整图像尺寸
        img_resized = image.resize((self.width, self.height), Image.LANCZOS)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_arr, 0)

        with self._graph.as_default():
            decoded = self._session.run(
                self._dec_out,
                feed_dict={self._dec_image: img_batch}
            )

        return decoded[0]

    def decode_from_path(self, image_path: str) -> np.ndarray:
        """从文件路径加载并解码"""
        image = Image.open(image_path).convert('RGB')
        return self.decode(image)

    def verify(self, image: Image.Image, expected_secret: np.ndarray,
               max_hamming: int = 10) -> Dict:
        """
        验证解码结果是否与期望的用户 ID 匹配
        使用汉明距离阈值，提高容错率

        Returns:
            {
                'bit_accuracy': float,
                'string_accuracy': bool,      # 精确匹配
                'hamming_match': bool,        # 汉明距离匹配（推荐）
                'hamming_distance': int,
                'max_allowed_hamming': int,
            }
        """
        decoded = self.decode(image)
        decoded_rounded = np.round(decoded).astype(int)
        expected_rounded = np.round(expected_secret).astype(int)

        bit_acc = float(np.mean(decoded_rounded == expected_rounded))
        str_acc = bool(np.array_equal(decoded_rounded, expected_rounded))
        hamming = int(np.sum(decoded_rounded != expected_rounded))

        # 汉明距离 <= max_hamming 即认为匹配
        hamming_match = hamming <= max_hamming

        return {
            'bit_accuracy': round(bit_acc, 4),
            'string_accuracy': str_acc,
            'hamming_match': hamming_match,
            'hamming_distance': hamming,
            'max_allowed_hamming': max_hamming,
        }

    def close(self):
        """关闭 TensorFlow 会话"""
        if self._session is not None:
            self._session.close()
            self._session = None
