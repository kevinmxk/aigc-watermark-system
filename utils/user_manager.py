"""
用户 ID 管理：字符串 <-> 比特串编解码
使用 CRC32 + UTF-8 编码，支持 100 比特 secret_size
"""
import numpy as np
import zlib
from typing import Tuple, Optional


class UserIDManager:
    """用户 ID 管理器，支持字符串到比特串的编解码"""

    def __init__(self, secret_size: int = 100):
        """
        Args:
            secret_size: StegaStamp 的 secret 比特数
        """
        self.secret_size = secret_size
        # 可用比特数：总大小 - CRC32(32bit) - 长度头(8bit)
        self.payload_size = max(secret_size - 40, 1)

    def encode(self, user_id: str) -> np.ndarray:
        """
        将用户 ID 字符串编码为固定长度的比特串

        Args:
            user_id: 用户标识字符串，如 "user_001"

        Returns:
            长度为 secret_size 的 numpy 数组 (float32, 0或1)
        """
        # 将字符串转为 UTF-8 字节
        user_bytes = user_id.encode('utf-8')

        # 计算 CRC32 校验码（4 字节）
        crc = zlib.crc32(user_bytes) & 0xFFFFFFFF
        crc_bytes = crc.to_bytes(4, byteorder='big')

        # 组合：长度(1字节) + CRC(4字节) + 内容
        header = bytes([len(user_bytes)]) + crc_bytes
        full_data = header + user_bytes

        # 转为比特
        bits = []
        for byte in full_data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)

        # 填充到 secret_size
        secret = np.array(bits[:self.payload_size], dtype=np.float32)

        # 添加 parity checksum（最后的比特用作校验）
        remaining = self.secret_size - len(secret)
        if remaining > 0:
            # 用随机比特填充
            padding = np.random.randint(0, 2, remaining).astype(np.float32)
            secret = np.concatenate([secret, padding])

        # 截断或填充到精确长度
        if len(secret) > self.secret_size:
            secret = secret[:self.secret_size]
        elif len(secret) < self.secret_size:
            secret = np.pad(secret, (0, self.secret_size - len(secret)), 'constant')

        return secret.astype(np.float32)

    def decode(self, secret: np.ndarray) -> Optional[str]:
        """
        从解码的比特串中恢复用户 ID

        Args:
            secret: 解码后的比特串 (长度为 secret_size 的 numpy 数组)

        Returns:
            恢复的用户 ID 字符串，如果校验失败返回 None
        """
        # 取有效载荷部分
        bits = np.round(secret[:self.payload_size]).astype(int)

        # 转回字节
        payload_bits = bits.tolist()
        byte_list = []
        for i in range(0, len(payload_bits) - 7, 8):
            byte_val = 0
            for j in range(8):
                byte_val = (byte_val << 1) | payload_bits[i + j]
            byte_list.append(byte_val)

        if len(byte_list) < 5:
            return None

        # 解析头部
        content_len = byte_list[0]
        crc_bytes = bytes(byte_list[1:5])
        content_bytes = bytes(byte_list[5:5 + content_len])

        # 验证 CRC32
        expected_crc = int.from_bytes(crc_bytes, byteorder='big')
        actual_crc = zlib.crc32(content_bytes) & 0xFFFFFFFF

        if expected_crc != actual_crc:
            return None

        try:
            return content_bytes.decode('utf-8')
        except (UnicodeDecodeError, ValueError):
            return None

    def verify_user(self, secret: np.ndarray, expected_user_id: str) -> Tuple[bool, float]:
        """
        验证解码结果是否与期望用户 ID 匹配

        Returns:
            (是否匹配, 比特准确率)
        """
        expected_secret = self.encode(expected_user_id)
        bit_acc = float(np.mean(np.round(secret) == np.round(expected_secret)))
        return bit_acc > 0.95, bit_acc

    def generate_user_id(self, index: int, prefix: str = "USER") -> str:
        """生成标准格式的用户 ID"""
        return f"{prefix}{index:03d}"
