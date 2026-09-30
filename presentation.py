"""Plain-language summaries use only supported estimates; no invented certainty."""
from return_first_model import finite_scalar
import math


def plain_summary(forecast: dict) -> str:
    f=forecast or {}
    if not (f.get('estimate_available') is True and f.get('sample_supported') is True):
        return '\u76f8\u4f3c\u6b77\u53f2\u8b49\u64da\u9084\u4e0d\u8db3\uff0c\u76ee\u524d\u53ea\u5217\u89c0\u5bdf\uff0c\u4e0d\u7d66\u9810\u671f\u5831\u916c\u627f\u8afe\u3002'
    s=f.get('strategy') or {}
    mean=finite_scalar(s.get('mean')); tail=finite_scalar(s.get('expected_shortfall10_loss'))
    prob=finite_scalar(s.get('p_profit'))
    if not all(math.isfinite(v) for v in (mean,tail,prob)):
        return '\u5831\u916c\u7d71\u8a08\u4e0d\u5b8c\u6574\uff0c\u8acb\u5148\u67e5\u770b\u8cc7\u6599\u8aaa\u660e\u3002'
    return (f'\u76f8\u4f3c\u6a23\u672c\u6263\u8a2d\u5b9a\u8cbb\u7528\u5f8c\uff0c\u5e73\u5747 {mean:+.1%}\uff1b'
            f'\u6a23\u672c\u7372\u5229\u6bd4\u4f8b {prob:.0%}\uff0c\u6700\u5dee10%\u7d50\u679c\u7684\u5e73\u5747\u640d\u5931 {tail:.1%}\u3002'
            '\u9019\u662f\u6b77\u53f2\u689d\u4ef6\u4f30\u8a08\uff0c\u4e0d\u662f\u672a\u4f86\u52dd\u7387\u6216\u6700\u5927\u640d\u5931\u4fdd\u8b49\u3002')
