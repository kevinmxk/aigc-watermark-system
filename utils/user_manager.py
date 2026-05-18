"""
用户 ID 管理：支持三种验证模式

1. 精确匹配 (Exact Match)
2. 汉明距离阈值匹配 (Hamming Threshold)
3. BCH 纠错 + CRC 校验 (BCH + CRC) [推荐]

BCH + CRC 方案（文档推荐）：
- short_id (16 bit) + CRC-16 (16 bit) = 32 bit 消息
- BCH 编码生成 ~63-100 bit 码字
- 可纠正 3~8 bit 错误
"""
import numpy as np
import zlib
import json
import os
from typing import Optional, Tuple, Dict, List
from pathlib import Path


class UserIDManager:
    """
    用户 ID 管理器 - 基础版（汉明距离验证）

    保留作为 baseline 对照组
    """

    def __init__(self, secret_size: int = 100, max_hamming: int = 10):
        """
        Args:
            secret_size: StegaStamp 的 secret 比特数
            max_hamming: 最大允许汉明距离
        """
        self.secret_size = secret_size
        self.max_hamming = max_hamming

    def encode(self, user_id: str) -> np.ndarray:
        """将用户 ID 编码为比特串"""
        user_bytes = user_id.encode('utf-8')
        crc = zlib.crc32(user_bytes) & 0xFFFFFFFF
        full_bytes = crc.to_bytes(4, byteorder='big') + user_bytes

        bits = []
        for byte in full_bytes:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)

        if len(bits) > self.secret_size:
            bits = bits[:self.secret_size]
        else:
            bits = bits + [0] * (self.secret_size - len(bits))

        return np.array(bits, dtype=np.float32)

    def decode(self, secret: np.ndarray) -> Optional[str]:
        """从比特串恢复用户 ID（简化版）"""
        return None  # 需要更复杂的解码逻辑

    def verify_user(self, secret: np.ndarray, expected_user_id: str) -> Tuple[bool, float, Dict]:
        """
        使用汉明距离验证用户 ID

        Returns:
            (是否匹配, 比特准确率, 详细信息)
        """
        expected_secret = self.encode(expected_user_id)
        actual = np.round(secret).astype(int)
        expected = np.round(expected_secret).astype(int)

        hamming = int(np.sum(actual != expected))
        bit_acc = float(np.mean(actual == expected))
        match = hamming <= self.max_hamming

        detail = {
            'method': 'hamming',
            'hamming_distance': hamming,
            'bit_accuracy': bit_acc,
            'max_allowed_hamming': self.max_hamming,
            'match': match,
        }

        return match, bit_acc, detail


class UserIDManagerBCH:
    """
    用户 ID 管理器 - BCH + CRC 增强版（文档推荐方案）

    流程：
    编码：user_id -> short_id -> CRC-16 -> BCH 编码 -> 100 bit
    解码：100 bit -> BCH 解码纠错 -> CRC 校验 -> short_id -> user_id
    """

    def __init__(self,
                 secret_size: int = 100,
                 short_id_bits: int = 16,
                 crc_bits: int = 16,
                 bch_n: int = 63,
                 bch_k: int = 32,
                 bch_t: int = 3,
                 mapping_file: str = None):
        """
        Args:
            secret_size: StegaStamp 总载荷 bit 数
            short_id_bits: short_id 位数 (推荐 16)
            crc_bits: CRC 位数 (推荐 16 或 8)
            bch_n: BCH 码字长度
            bch_k: BCH 消息长度 (short_id_bits + crc_bits)
            bch_t: BCH 可纠正错误数
            mapping_file: 用户映射表文件路径 (默认使用项目根目录下的 data/user_mapping.json)
        """
        self.secret_size = secret_size
        self.short_id_bits = short_id_bits
        self.crc_bits = crc_bits
        self.bch_n = bch_n
        self.bch_k = bch_k
        self.bch_t = bch_t

        # 使用固定路径：项目根目录下的 data/user_mapping.json
        if mapping_file is None:
            # 获取项目根目录 (utils/..)
            project_root = Path(__file__).parent.parent.absolute()
            data_dir = project_root / "data"
            data_dir.mkdir(exist_ok=True)
            self.mapping_file = str(data_dir / "user_mapping.json")
        else:
            self.mapping_file = mapping_file

        # 用户映射表
        self._user_to_short: Dict[str, int] = {}
        self._short_to_user: Dict[int, str] = {}
        self._next_short_id = 1

        # 加载映射表
        self._load_mapping()

        # 初始化 BCH 和 CRC
        self._init_codec()

    def _init_codec(self):
        """初始化 BCH 和 CRC 编解码器"""
        try:
            from .crc import StegaStampCRC
            from .bch_codec import BCHCodec

            self.crc_codec = StegaStampCRC(
                short_id_bits=self.short_id_bits,
                crc_bits=self.crc_bits
            )
            self.bch_codec = BCHCodec(
                n=self.bch_n,
                k=self.bch_k,
                t=self.bch_t
            )
            self.codec_available = True
        except ImportError as e:
            print(f"警告: BCH/CRC 库未安装: {e}")
            self.codec_available = False
            self.crc_codec = None
            self.bch_codec = None

    def _load_mapping(self):
        """加载用户映射表"""
        if os.path.exists(self.mapping_file):
            with open(self.mapping_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._user_to_short = {k: int(v) for k, v in data.get('user_to_short', {}).items()}
                self._short_to_user = {int(k): v for k, v in data.get('short_to_user', {}).items()}
                self._next_short_id = data.get('next_short_id', 1)

    def _save_mapping(self):
        """保存用户映射表"""
        data = {
            'user_to_short': self._user_to_short,
            'short_to_user': {str(k): v for k, v in self._short_to_user.items()},
            'next_short_id': self._next_short_id,
        }
        os.makedirs(os.path.dirname(self.mapping_file) or '.', exist_ok=True)
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def register_user(self, user_id: str) -> int:
        """
        注册用户，分配 short_id

        Args:
            user_id: 用户标识字符串

        Returns:
            分配的 short_id
        """
        if user_id in self._user_to_short:
            return self._user_to_short[user_id]

        short_id = self._next_short_id
        self._next_short_id += 1

        self._user_to_short[user_id] = short_id
        self._short_to_user[short_id] = user_id

        self._save_mapping()
        return short_id

    def encode(self, user_id: str) -> np.ndarray:
        """
        编码用户 ID 为比特串（BCH + CRC 方案）

        流程：
        user_id -> short_id -> CRC -> BCH 编码 -> 填充到 secret_size
        """
        if not self.codec_available:
            # 降级为旧方案
            return self._encode_legacy(user_id)

        # 获取或分配 short_id
        short_id = self.register_user(user_id)

        # CRC 编码: short_id -> short_bits + crc_bits
        crc_codeword = self.crc_codec.encode(short_id)
        # crc_codeword 是 short_id_bits + crc_bits = 32 bit

        # BCH 编码
        bch_codeword = self.bch_codec.encode(crc_codeword)
        # bch_codeword 是 bch_n bit (约 63 bit)

        # 填充到 secret_size
        final_bits = self._pad_to_secret_size(bch_codeword)

        return final_bits.astype(np.float32)

    def _encode_legacy(self, user_id: str) -> np.ndarray:
        """降级方案 - 复用 UserIDManager.encode()"""
        # 创建临时的旧版管理器来复用其编码逻辑
        legacy_manager = UserIDManager(secret_size=self.secret_size)
        return legacy_manager.encode(user_id)
                bits.append((byte >> i) & 1)

        if len(bits) > self.secret_size:
            bits = bits[:self.secret_size]
        else:
            bits = bits + [0] * (self.secret_size - len(bits))

        return np.array(bits, dtype=np.float32)

    def _pad_to_secret_size(self, bits: np.ndarray) -> np.ndarray:
        """填充或截断到 secret_size"""
        bits = bits.flatten()
        if len(bits) >= self.secret_size:
            return bits[:self.secret_size]
        else:
            return np.pad(bits, (0, self.secret_size - len(bits)), 'constant')

    def decode(self, secret: np.ndarray) -> Tuple[bool, Optional[str], Dict]:
        """
        解码比特串为用户 ID

        流程：
        secret -> BCH 解码纠错 -> CRC 校验 -> short_id -> user_id

        Returns:
            (成功标志, 用户ID, 详细信息)
        """
        if not self.codec_available:
            return False, None, {'error': 'BCH codec not available'}

        bits = np.round(secret).astype(np.uint8)

        # 提取有效 BCH 码字
        bch_codeword = bits[:self.bch_n]

        # BCH 解码
        bch_success, crc_codeword = self.bch_codec.decode(bch_codeword)
        if not bch_success:
            return False, None, {
                'stage': 'bch',
                'error': 'BCH decoding failed',
                'extracted_bits': bits[:20].tolist(),
            }

        # CRC 解码
        crc_success, short_id = self.crc_codec.decode(crc_codeword)
        if not crc_success:
            return False, None, {
                'stage': 'crc',
                'error': 'CRC verification failed',
            }

        # 映射回 user_id
        user_id = self._short_to_user.get(short_id)
        if user_id is None:
            return False, None, {
                'stage': 'mapping',
                'error': f'Unknown short_id: {short_id}',
            }

        return True, user_id, {
            'stage': 'success',
            'short_id': short_id,
            'method': 'bch_crc',
        }

    def verify_user(self, secret: np.ndarray, expected_user_id: str) -> Tuple[bool, float, Dict]:
        """
        验证解码结果是否与期望用户 ID 匹配

        Returns:
            (是否匹配, 置信度, 详细信息)
        """
        success, decoded_id, detail = self.decode(secret)

        if not success:
            # BCH+CRC 失败，返回详细信息
            return False, 0.0, {
                **detail,
                'expected_user': expected_user_id,
                'method': 'bch_crc',
            }

        match = (decoded_id == expected_user_id)

        return match, 1.0 if match else 0.0, {
            'method': 'bch_crc',
            'decoded_user': decoded_id,
            'expected_user': expected_user_id,
            'match': match,
            **detail,
        }

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            'total_users': len(self._user_to_short),
            'next_short_id': self._next_short_id,
            'short_id_bits': self.short_id_bits,
            'crc_bits': self.crc_bits,
            'bch_params': {
                'n': self.bch_n,
                'k': self.bch_k,
                't': self.bch_t,
            },
            'codec_available': self.codec_available,
        }


class HybridUserIDManager:
    """
    混合用户 ID 管理器

    支持三种验证方式：
    1. Exact Match: 精确匹配
    2. Hamming: 汉明距离阈值匹配
    3. BCH_CRC: BCH 纠错 + CRC 校验（推荐）
    """

    def __init__(self,
                 secret_size: int = 100,
                 mode: str = 'bch_crc',
                 max_hamming: int = 10,
                 mapping_file: str = "user_mapping.json"):
        """
        Args:
            secret_size: StegaStamp 载荷 bit 数
            mode: 验证模式 ('exact', 'hamming', 'bch_crc')
            max_hamming: 汉明距离阈值
            mapping_file: 用户映射表文件
        """
        self.mode = mode
        self.secret_size = secret_size

        # 初始化各种管理器
        self.exact_manager = UserIDManager(secret_size, max_hamming=0)
        self.hamming_manager = UserIDManager(secret_size, max_hamming)
        self.bch_manager = UserIDManagerBCH(secret_size, mapping_file=mapping_file)

    def encode(self, user_id: str) -> np.ndarray:
        """编码用户 ID"""
        if self.mode == 'bch_crc':
            return self.bch_manager.encode(user_id)
        else:
            return self.hamming_manager.encode(user_id)

    def verify(self, secret: np.ndarray, expected_user_id: str) -> Tuple[bool, float, Dict]:
        """验证用户 ID"""
        if self.mode == 'exact':
            match, acc, detail = self.exact_manager.verify_user(secret, expected_user_id)
            detail['mode'] = 'exact'
            return match, acc, detail

        elif self.mode == 'hamming':
            match, acc, detail = self.hamming_manager.verify_user(secret, expected_user_id)
            detail['mode'] = 'hamming'
            return match, acc, detail

        elif self.mode == 'bch_crc':
            return self.bch_manager.verify_user(secret, expected_user_id)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def compare_all_modes(self, secret: np.ndarray, expected_user_id: str) -> Dict:
        """
        比较所有验证模式的结果（用于实验对比）

        Returns:
            各模式的验证结果
        """
        results = {}

        # 精确匹配
        match, acc, detail = self.exact_manager.verify_user(secret, expected_user_id)
        results['exact'] = {'match': match, 'accuracy': acc, 'detail': detail}

        # 汉明距离
        match, acc, detail = self.hamming_manager.verify_user(secret, expected_user_id)
        results['hamming'] = {'match': match, 'accuracy': acc, 'detail': detail}

        # BCH + CRC
        match, acc, detail = self.bch_manager.verify_user(secret, expected_user_id)
        results['bch_crc'] = {'match': match, 'accuracy': acc, 'detail': detail}

        return results


def test_user_manager():
    """测试用户管理器"""
    print("=" * 60)
    print("用户 ID 管理器测试")
    print("=" * 60)

    # 测试汉明距离方案
    print("\n--- 汉明距离方案 ---")
    um = UserIDManager(secret_size=100, max_hamming=5)
    user_id = "user_001"
    secret = um.encode(user_id)
    print(f"用户: {user_id}")
    print(f"编码后 ({len(secret)} bit): {secret[:10]}...")

    # 验证
    match, acc, detail = um.verify_user(secret, user_id)
    print(f"精确验证: {'通过' if match else '失败'}, 准确率: {acc:.4f}")

    # 添加错误
    corrupted = secret.copy()
    corrupted[5:10] = 1 - corrupted[5:10]
    match, acc, detail = um.verify_user(corrupted, user_id)
    print(f"5 bit 错误后验证: {'通过' if match else '失败'}, "
          f"汉明距离: {detail['hamming_distance']}, 准确率: {acc:.4f}")

    # 测试 BCH + CRC 方案
    print("\n--- BCH + CRC 方案 ---")
    um_bch = UserIDManagerBCH(secret_size=100)
    user_id = "user_002"
    secret = um_bch.encode(user_id)
    print(f"用户: {user_id}")
    print(f"编码后 ({len(secret)} bit)")

    # 解码验证
    success, decoded, detail = um_bch.decode(secret)
    print(f"解码: {'成功' if success else '失败'}, 用户: {decoded}")

    if success:
        # 添加错误测试
        print("\n错误恢复测试:")
        for num_errors in [3, 5, 10]:
            corrupted = secret.copy()
            indices = np.random.choice(100, num_errors, replace=False)
            corrupted[indices] = 1 - corrupted[indices]

            success, decoded, detail = um_bch.decode(corrupted)
            print(f"  {num_errors} bit 错误: 解码{'成功' if success else '失败'}")

    # 测试混合管理器
    print("\n--- 混合管理器对比 ---")
    hybrid = HybridUserIDManager(mode='bch_crc')
    user_id = "user_003"
    secret = hybrid.encode(user_id)

    results = hybrid.compare_all_modes(secret, user_id)
    for mode, result in results.items():
        print(f"  {mode}: {'通过' if result['match'] else '失败'}, "
              f"准确率: {result['accuracy']:.4f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    test_user_manager()
