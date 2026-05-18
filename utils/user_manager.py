"""
用户 ID 管理：字符串 <-> 比特串编解码
使用汉明距离 + 阈值匹配，提高容错率
"""
import numpy as np
import zlib
from typing import Optional, Tuple, Dict, List


class UserIDManager:
    """用户 ID 管理器，使用汉明距离验证"""

    def __init__(self, secret_size: int = 100, max_hamming: int = 10):
        """
        Args:
            secret_size: StegaStamp 的 secret 比特数
            max_hamming: 最大允许汉明距离（100 比特中最多错几个比特）
        """
        self.secret_size = secret_size
        self.max_hamming = max_hamming

    def encode(self, user_id: str) -> np.ndarray:
        """
        将用户 ID 编码为固定长度的比特串

        方案：CRC32 + UTF-8 编码
        """
        # 将字符串转为 UTF-8 字节 + CRC32
        user_bytes = user_id.encode('utf-8')
        crc = zlib.crc32(user_bytes) & 0xFFFFFFFF
        full_bytes = crc.to_bytes(4, byteorder='big') + user_bytes

        # 转为比特
        bits = []
        for byte in full_bytes:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)

        # 填充/截断到 secret_size
        if len(bits) > self.secret_size:
            bits = bits[:self.secret_size]
        else:
            bits = bits + [0] * (self.secret_size - len(bits))

        return np.array(bits, dtype=np.float32)

    def decode(self, secret: np.ndarray) -> Optional[str]:
        """
        从比特串中恢复用户 ID（简化版，仅当精确匹配时有效）
        推荐使用 verify_user() 进行容错验证
        """
        bits = np.round(secret[:32]).astype(int).tolist()
        if len(bits) < 32:
            return None

        # 提取 CRC 和内容
        crc_bytes = bytes([sum(bits[i*8:(i+1)*8] << (7-j) for j in range(8)) for i in range(4)])
        return None  # 需要更复杂的解码逻辑

    def verify_user(self, secret: np.ndarray, expected_user_id: str) -> Tuple[bool, float, Dict]:
        """
        验证解码结果是否与期望用户 ID 匹配

        使用汉明距离：100 比特中最多允许 max_hamming 个比特错误

        Returns:
            (是否匹配, 比特准确率, 详细信息)
        """
        expected_secret = self.encode(expected_user_id)
        actual = np.round(secret).astype(int)
        expected = np.round(expected_secret).astype(int)

        hamming = int(np.sum(actual != expected))
        bit_acc = float(np.mean(actual == expected))

        # 汉明距离 <= max_hamming 即认为匹配
        match = hamming <= self.max_hamming

        detail = {
            'hamming_distance': hamming,
            'bit_accuracy': bit_acc,
            'max_allowed_hamming': self.max_hamming,
            'match': match,
        }

        return match, bit_acc, detail

    def identify_user(self, secret: np.ndarray, known_users: List[str]) -> Tuple[Optional[str], float, Dict]:
        """
        在已知用户列表中识别最匹配的用户

        Returns:
            (匹配的用户ID, 比特准确率, 详细信息)
        """
        best_user = None
        best_hamming = self.secret_size + 1
        best_detail = {}

        for user_id in known_users:
            expected = self.encode(user_id)
            actual = np.round(secret).astype(int)
            expected_r = np.round(expected).astype(int)
            hamming = int(np.sum(actual != expected_r))
            bit_acc = 1.0 - hamming / self.secret_size

            if hamming < best_hamming:
                best_hamming = hamming
                best_user = user_id
                best_detail = {
                    'hamming_distance': hamming,
                    'bit_accuracy': bit_acc,
                }

        if best_hamming <= self.max_hamming:
            return best_user, best_detail['bit_accuracy'], best_detail
        return None, best_detail.get('bit_accuracy', 0), best_detail

    def generate_user_id(self, index: int, prefix: str = "USER") -> str:
        """生成标准格式用户 ID"""
        return f"{prefix}{index:03d}"
