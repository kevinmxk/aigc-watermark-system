"""
BCH 纠错码编解码模块 - 使用真正的 BCH 库

依赖:
- bchlib: 真正的 BCH 纠错码实现
- reedsolo: Reed-Solomon 码（备选）

安装:
    pip install bchlib
    # 或备选
    pip install reedsolo
"""
import numpy as np
from typing import Tuple, Optional
import math
import warnings


class BCHCodec:
    """
    BCH 纠错码编解码器 - 基于 bchlib

    BCH 码参数:
    - n: 码字长度 (总 bit 数)
    - k: 消息长度 (原始消息 bit 数)
    - t: 可纠正错误数

    对于 StegaStamp (100 bit 载荷)，推荐:
    - k=32 (16 bit short_id + 16 bit CRC)
    - n=63 (63 bit 码字，约可纠正 3-5 bit)
    """

    def __init__(self, n: int = 63, k: int = 32, t: int = 3):
        """
        初始化 BCH 编解码器

        Args:
            n: 码字长度 (总 bit 数)，必须是 2^m - 1 形式
            k: 消息长度 (原始消息 bit 数)
            t: 可纠正错误数
        """
        self.n = n
        self.k = k
        self.t = t

        # 尝试使用真正的 BCH 库
        self._init_bchlib()

        if not self.available:
            warnings.warn(
                "bchlib 未安装，使用 Reed-Solomon 备选方案。\n"
                "安装 bchlib: pip install bchlib\n"
                "这将提供更好的纠错性能。"
            )
            self._init_reedsolo()

    def _init_bchlib(self):
        """使用 bchlib 库初始化"""
        try:
            import bchlib

            # bchlib 需要消息字节数
            # k=32 bit = 4 bytes
            self.bch = bchlib.BCH(32, t=self.t)  # 32 bytes 消息长度
            self.available = True
            self.backend = 'bchlib'
        except ImportError:
            self.bch = None
            self.available = False

    def _init_reedsolo(self):
        """使用 reedsolo 作为备选"""
        try:
            from reedsolo import RSCodec

            # RS 参数: n=255, k=223 (经典参数)
            # 但为了 fit 100 bit，使用更小参数
            # 消息 32 bit = 4 bytes, 纠错 10 bytes (80 bit)
            self.rs = RSCodec(nsym=10, nsize=255)
            self.rs_available = True
            self.backend = 'reedsolo'
        except ImportError:
            self.rs = None
            self.rs_available = False
            self.backend = 'simple'

    def encode(self, message_bits: np.ndarray) -> np.ndarray:
        """
        BCH 编码

        Args:
            message_bits: 原始消息 (k bit)

        Returns:
            BCH 码字 (n bit)
        """
        # 确保消息长度正确
        message_bits = self._pad_bits(message_bits, self.k)

        if self.backend == 'bchlib':
            return self._encode_bchlib(message_bits)
        elif self.backend == 'reedsolo':
            return self._encode_reedsolo(message_bits)
        else:
            return self._encode_simple(message_bits)

    def decode(self, received_bits: np.ndarray) -> Tuple[bool, np.ndarray]:
        """
        BCH 解码并纠错

        Args:
            received_bits: 接收到的码字 (n bit)

        Returns:
            (成功标志, 恢复的消息 (k bit))
        """
        if len(received_bits) < self.n:
            received_bits = self._pad_bits(received_bits, self.n)

        if self.backend == 'bchlib':
            return self._decode_bchlib(received_bits)
        elif self.backend == 'reedsolo':
            return self._decode_reedsolo(received_bits)
        else:
            return self._decode_simple(received_bits)

    def _pad_bits(self, bits: np.ndarray, target_len: int) -> np.ndarray:
        """填充或截断 bit 数组到目标长度"""
        bits = bits.flatten()
        if len(bits) >= target_len:
            return bits[:target_len]
        else:
            return np.pad(bits, (0, target_len - len(bits)), 'constant')

    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """bit 数组转字节"""
        bits = bits.flatten()
        n_bytes = (len(bits) + 7) // 8
        result = bytearray(n_bytes)
        for i, bit in enumerate(bits):
            if bit:
                result[i // 8] |= (1 << (7 - i % 8))
        return bytes(result)

    def _bytes_to_bits(self, data: bytes, n_bits: int) -> np.ndarray:
        """字节转 bit 数组"""
        bits = []
        for byte in data:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return np.array(bits[:n_bits], dtype=np.uint8)

    def _encode_bchlib(self, message_bits: np.ndarray) -> np.ndarray:
        """使用 bchlib 编码"""
        import bchlib

        # bit 转 bytes
        message_bytes = self._bits_to_bytes(message_bits)

        # 编码
        # bchlib 返回 (ecc_bytes, ecc_len)
        ecc = self.bch.encode(message_bytes)

        # 组合: message + ecc
        codeword_bytes = message_bytes + ecc

        # bytes 转 bit
        codeword_bits = self._bytes_to_bits(codeword_bytes, self.n)

        return codeword_bits

    def _decode_bchlib(self, received_bits: np.ndarray) -> Tuple[bool, np.ndarray]:
        """使用 bchlib 解码并纠错"""
        import bchlib

        # bit 转 bytes
        received_bytes = self._bits_to_bytes(received_bits)

        # bchlib 需要分开 data 和 ecc
        # 假设前 k/8 字节是消息，后面是 ecc
        data_len = (self.k + 7) // 8
        data = received_bytes[:data_len]
        ecc = received_bytes[data_len:]

        # 解码纠错
        try:
            # bchlib 的 decode 返回 (corrected_data, bitflips)
            corrected = self.bch.decode(data, ecc)

            if corrected is not None:
                # 成功纠错
                corrected_bits = self._bytes_to_bits(corrected, self.k)
                return True, corrected_bits
            else:
                # 纠错失败（错误太多）
                return False, self._bytes_to_bits(data, self.k)
        except Exception as e:
            return False, self._bytes_to_bits(data, self.k)

    def _encode_reedsolo(self, message_bits: np.ndarray) -> np.ndarray:
        """使用 reedsolo 编码"""
        # bit 转 bytes
        message_bytes = self._bits_to_bytes(message_bits)

        # RS 编码
        codeword_bytes = self.rs.encode(message_bytes)

        # bytes 转 bit
        codeword_bits = self._bytes_to_bits(codeword_bytes, self.n)

        return codeword_bits

    def _decode_reedsolo(self, received_bits: np.ndarray) -> Tuple[bool, np.ndarray]:
        """使用 reedsolo 解码并纠错"""
        # bit 转 bytes
        received_bytes = self._bits_to_bytes(received_bits)

        try:
            # RS 解码
            decoded_bytes = self.rs.decode(received_bytes)

            # bytes 转 bit
            decoded_bits = self._bytes_to_bits(decoded_bytes, self.k)
            return True, decoded_bits
        except Exception as e:
            # 解码失败
            return False, received_bits[:self.k]

    def _encode_simple(self, message_bits: np.ndarray) -> np.ndarray:
        """
        降级方案：简单重复码 + XOR校验

        注意：这不是真正的BCH，仅作为备选
        """
        redundancy = self.n - self.k
        checksum_bits = self._compute_checksum(message_bits, redundancy)
        codeword = np.concatenate([message_bits, checksum_bits])

        while len(codeword) < self.n:
            codeword = np.concatenate([codeword, message_bits])

        return codeword[:self.n]

    def _decode_simple(self, received_bits: np.ndarray) -> Tuple[bool, np.ndarray]:
        """降级方案：简单解码"""
        message_part = received_bits[:self.k].copy()
        return True, message_part

    def _compute_checksum(self, message_bits: np.ndarray, checksum_len: int) -> np.ndarray:
        """计算校验位（简化版）"""
        checksum = []
        step = max(1, len(message_bits) // checksum_len)

        for i in range(checksum_len):
            start = i * step
            end = min((i + 1) * step, len(message_bits))
            if start < len(message_bits):
                xor_val = np.bitwise_xor.reduce(message_bits[start:end].astype(int))
                checksum.append(xor_val % 2)
            else:
                checksum.append(0)

        return np.array(checksum, dtype=np.uint8)

    def get_params(self) -> dict:
        """获取 BCH 参数"""
        return {
            'n': self.n,
            'k': self.k,
            't': self.t,
            'backend': self.backend,
            'redundancy_bits': self.n - self.k,
            'available': self.available,
        }


def create_bch_for_stegastamp(
    short_id_bits: int = 16,
    crc_bits: int = 16,
    total_bits: int = 100,
) -> BCHCodec:
    """
    为 StegaStamp 创建合适的 BCH 编解码器

    Args:
        short_id_bits: short_id 位数 (默认 16)
        crc_bits: CRC 位数 (默认 16)
        total_bits: StegaStamp 总载荷 (默认 100)

    Returns:
        配置好的 BCHCodec
    """
    k = short_id_bits + crc_bits  # 原始消息长度

    # 推荐配置：n=63, k=32, t=3 (可纠正 3 bit)
    if total_bits >= 63 and k <= 32:
        return BCHCodec(n=63, k=k, t=3)
    elif total_bits >= 127 and k <= 64:
        return BCHCodec(n=127, k=k, t=5)
    else:
        n = min(total_bits, k * 2)
        return BCHCodec(n=n, k=k, t=2)


def test_bch():
    """测试 BCH 编解码"""
    print("=" * 60)
    print("BCH 编解码测试")
    print("=" * 60)

    # 创建 BCH 编解码器
    bch = BCHCodec(n=63, k=32, t=3)
    print(f"\n参数: {bch.get_params()}")

    if not bch.available:
        print("\n警告: 未安装 bchlib 或 reedsolo")
        print("安装命令: pip install bchlib reedsolo")
        print("当前使用降级方案（无纠错能力）")

    # 测试消息 (32 bit)
    message = np.array([1, 0, 1, 0] * 8, dtype=np.uint8)
    print(f"\n原始消息 ({len(message)} bit): {message[:16]}...")

    # 编码
    codeword = bch.encode(message)
    print(f"编码后 ({len(codeword)} bit)")

    # 添加错误测试
    print(f"\n错误恢复测试:")
    for num_errors in [1, 2, 3, 5, 10]:
        corrupted = codeword.copy()
        indices = np.random.choice(len(codeword), num_errors, replace=False)
        corrupted[indices] = 1 - corrupted[indices]

        success, recovered = bch.decode(corrupted)
        correct = np.array_equal(recovered, message) if success else False

        status = "✓" if correct else ("△" if success else "✗")
        print(f"  {status} {num_errors} bit 错误: 解码{'成功' if success else '失败'}, "
              f"恢复正确: {correct}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_bch()
