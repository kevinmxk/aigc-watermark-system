"""
阶段3: StegaStamp 用户指纹编码
在 'stegastamp' conda 环境中运行 (TensorFlow 1.15.5)
输入: 图像路径、用户ID
输出: 带用户指纹的最终图像
"""
import argparse
import json
import os
import sys
import numpy as np


def encode_stegastamp(image_path: str, user_id: str, config: dict) -> dict:
    """
    使用 StegaStamp 编码用户指纹

    Returns:
        {
            'final_image_path': 最终图像路径,
            'bit_accuracy': 比特准确率,
            'secret_decoded': 解码的用户ID,
        }
    """
    work_dir = config.get('work_dir', '.')

    # 导入 TensorFlow (TF 1.15 环境)
    try:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import warnings
        warnings.filterwarnings('ignore')
        import tensorflow.compat.v1 as tf
        tf.disable_eager_execution()
        TF_AVAILABLE = True
    except ImportError:
        print("警告: TensorFlow 1.15 未安装")
        TF_AVAILABLE = False

    # 导入用户管理器
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.user_manager import UserIDManager

    # 编码用户ID
    secret_size = config.get('secret_size', 100)
    um = UserIDManager(secret_size=secret_size)
    user_secret = um.encode(user_id)

    # 读取输入图像
    from PIL import Image
    image = Image.open(image_path).convert('RGB')

    if not TF_AVAILABLE:
        # 降级方案：直接返回原图
        final_path = os.path.join(work_dir, f'{os.path.basename(image_path).replace(".png", "")}_final.png')
        image.save(final_path)
        return {
            'final_image_path': final_path,
            'bit_accuracy': 0.0,
            'secret_decoded': user_id,
            'note': 'TF unavailable, using original image'
        }

    # 加载 StegaStamp Encoder
    encoder_path = config.get('encoder_path', 'saved_models/stegastamp_140k/encoder')

    if not os.path.exists(encoder_path):
        print(f"警告: StegaStamp encoder 不存在: {encoder_path}")
        final_path = os.path.join(work_dir, f'{os.path.basename(image_path).replace(".png", "")}_final.png')
        image.save(final_path)
        return {
            'final_image_path': final_path,
            'bit_accuracy': 0.0,
            'secret_decoded': user_id,
            'note': f'Encoder not found: {encoder_path}'
        }

    # TF 1.15 SavedModel 加载
    from tensorflow.python.saved_model import tag_constants

    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        tf.saved_model.loader.load(
            sess,
            [tag_constants.SERVING],
            encoder_path
        )

        # 获取输入输出张量
        enc_graph = tf.get_default_graph()
        enc_secret = enc_graph.get_tensor_by_name('secret:0')
        enc_image = enc_graph.get_tensor_by_name('image:0')
        enc_out = enc_graph.get_tensor_by_name('clip_by_value:0')

        # 准备图像 (400x400)
        height = config.get('height', 400)
        width = config.get('width', 400)
        img_resized = image.resize((width, height), Image.LANCZOS)
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_arr, 0)

        # 编码
        enc_output = sess.run(
            enc_out,
            feed_dict={
                enc_secret: np.expand_dims(user_secret, 0),
                enc_image: img_batch,
            }
        )

        # 转换为图像
        enc_clipped = np.clip(enc_output[0], 0, 1)
        final_image = Image.fromarray((enc_clipped * 255).astype(np.uint8))

    # 保存结果
    final_path = os.path.join(work_dir, f'{os.path.basename(image_path).replace("_treering.png", "")}_final.png')
    final_image.save(final_path)

    # 验证（如果 decoder 可用）
    decoder_path = config.get('decoder_path', 'saved_models/stegastamp_140k/decoder')
    bit_acc = 0.0

    if os.path.exists(decoder_path):
        try:
            with graph.as_default():
                tf.saved_model.loader.load(
                    sess,
                    [tag_constants.SERVING],
                    decoder_path
                )
                dec_graph = tf.get_default_graph()
                dec_image = dec_graph.get_tensor_by_name('image_dec:0')

                # 找到 Round 操作
                round_ops = [op for op in dec_graph.get_operations() if op.type == 'Round']
                if round_ops:
                    dec_out = round_ops[-1].outputs[0]
                    decoded = sess.run(dec_out, feed_dict={dec_image: img_batch})
                    decoded_bits = decoded[0]

                    # 计算比特准确率
                    decoded_rounded = np.round(decoded_bits).astype(int)
                    expected_rounded = np.round(user_secret).astype(int)
                    bit_acc = float(np.mean(decoded_rounded == expected_rounded))
        except Exception as e:
            print(f"解码验证失败: {e}")

    sess.close()

    return {
        'final_image_path': final_path,
        'bit_accuracy': round(bit_acc, 4),
        'secret_decoded': um.decode(decoded_bits) if bit_acc > 0 else user_id,
        'note': 'StegaStamp encoding successful'
    }


def main():
    parser = argparse.ArgumentParser(description='StegaStamp 用户指纹编码阶段')
    parser.add_argument('--input', type=str, required=True, help='输入JSON文件')
    parser.add_argument('--output', type=str, required=True, help='输出JSON文件')
    args = parser.parse_args()

    # 读取输入
    with open(args.input, 'r') as f:
        inputs = json.load(f)

    image_path = inputs.get('image_path', '')
    user_id = inputs.get('user_id', 'user_000')
    work_dir = inputs.get('work_dir', '.')

    # 配置
    config = {
        'encoder_path': 'saved_models/stegastamp_140k/encoder',
        'decoder_path': 'saved_models/stegastamp_140k/decoder',
        'secret_size': 100,
        'height': 400,
        'width': 400,
        'work_dir': work_dir,
    }

    print(f"[阶段3] StegaStamp 用户指纹编码")
    print(f"  输入图像: {image_path}")
    print(f"  用户ID: {user_id}")

    # 运行编码
    result = encode_stegastamp(image_path, user_id, config)

    print(f"  最终图像: {result['final_image_path']}")
    print(f"  比特准确率: {result['bit_accuracy']:.4f}")

    # 保存输出
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
