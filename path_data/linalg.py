"""纯标准库的小型线性代数工具（分析规模很小，不需要外部依赖）。"""
import math


def zeros(n):
    return [0.0] * n


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mat_vec(mat, vec):
    return [dot(row, vec) for row in mat]


def solve(a, b):
    """高斯-约当消元解方阵 a x = b；奇异时返回 None。"""
    n = len(a)
    aug = [list(map(float, a[i])) + [float(b[i])] for i in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        aug[col] = [v / pv for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor:
                aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col])]
    return [aug[i][n] for i in range(n)]


def lstsq(mat, rhs, ridge=1e-9):
    """最小二乘（正规方程 + 极小岭项）；返回系数，奇异/空时返回 None。"""
    if not mat or not mat[0]:
        return None
    n = len(mat[0])
    ata = [[0.0] * n for _ in range(n)]
    atb = [0.0] * n
    for row, y in zip(mat, rhs):
        for i in range(n):
            if row[i] == 0:
                continue
            atb[i] += row[i] * y
            for j in range(n):
                ata[i][j] += row[i] * row[j]
    scale = max([ata[i][i] for i in range(n)] or [1.0])
    for i in range(n):
        ata[i][i] += ridge * max(scale, 1.0)
    return solve(ata, atb)


def rank(mat, tol=1e-9):
    if not mat or not mat[0]:
        return 0
    a = [list(map(float, row)) for row in mat]
    rows = len(a)
    cols = len(a[0])
    r = 0
    scale = max((abs(v) for row in a for v in row), default=1.0) or 1.0
    for col in range(cols):
        pivot = max(range(r, rows), key=lambda i: abs(a[i][col]))
        if abs(a[pivot][col]) <= tol * scale:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        pv = a[r][col]
        a[r] = [v / pv for v in a[r]]
        for i in range(rows):
            if i != r and a[i][col]:
                f = a[i][col]
                a[i] = [x - f * y for x, y in zip(a[i], a[r])]
        r += 1
        if r == rows:
            break
    return r


def nullspace(mat, tol=1e-9):
    """矩阵的零空间基（输入 m×n，返回 n 维向量列表）。"""
    if not mat or not mat[0]:
        return []
    a = [list(map(float, row)) for row in mat]
    rows = len(a)
    cols = len(a[0])
    pivots = []
    r = 0
    scale = max((abs(v) for row in a for v in row), default=1.0) or 1.0
    for col in range(cols):
        pivot = max(range(r, rows), key=lambda i: abs(a[i][col]))
        if abs(a[pivot][col]) <= tol * scale:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        pv = a[r][col]
        a[r] = [v / pv for v in a[r]]
        for i in range(rows):
            if i != r and a[i][col]:
                f = a[i][col]
                a[i] = [x - f * y for x, y in zip(a[i], a[r])]
        pivots.append(col)
        r += 1
        if r == rows:
            break
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        vec = zeros(cols)
        vec[f] = 1.0
        for i, p in enumerate(pivots):
            vec[p] = -a[i][f]
        basis.append(vec)
    return basis


def _jacobi(a, max_sweeps=100, tol=1e-12):
    """对称矩阵特征分解（Jacobi 旋转）：返回 (特征值, 特征向量按列)。"""
    n = len(a)
    a = [list(map(float, row)) for row in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = max((abs(a[i][j]) for i in range(n) for j in range(n) if i != j),
                  default=0.0)
        scale = max((abs(a[i][i]) for i in range(n)), default=1.0) or 1.0
        if off <= tol * scale:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = ((1.0 if theta >= 0 else -1.0)
                     / (abs(theta) + math.sqrt(theta * theta + 1.0)))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v


def pseudoinverse_diagonal(mat, tol=1e-9):
    """对称半正定矩阵 Moore-Penrose 伪逆的对角线，以及零空间基。

    对角元用于近似方差；零空间基用于标记不可单独识别的系数
    （这些系数的标准误无界，报告为 None 而不是给一个虚假的小数）。
    """
    if not mat:
        return [], []
    n = len(mat)
    sym = [[0.5 * (mat[i][j] + mat[j][i]) for j in range(n)] for i in range(n)]
    eigvals, eigvecs = _jacobi(sym)
    scale = max((abs(e) for e in eigvals), default=1.0) or 1.0
    diag = [0.0] * n
    null_basis = []
    for k, ev in enumerate(eigvals):
        vec = [eigvecs[i][k] for i in range(n)]
        if ev > tol * scale:
            for i in range(n):
                diag[i] += vec[i] * vec[i] / ev
        else:
            null_basis.append(vec)
    return diag, null_basis
