"""stem-synth: 180 synthetic math and physics problems in 12 tiers, from single-digit arithmetic to
calculus, complex analysis and electromagnetism.

Each problem has one correct answer and three distractors, built from common mistakes where one exists
(kinetic energy without the 1/2, a determinant with the cofactor signs dropped, i^2 = +1), otherwise near
misses. Asked in all three modes: pick the answer (choice), is this shown answer correct (noul), and which
order of magnitude is |answer| in (score).
"""

import random
from fractions import Fraction

ID = "stem-synth"
TITLE = "STEM (synthetic)"
BLURB = ("180 generated problems in 12 tiers, from single-digit arithmetic to calculus, complex analysis and "
         "electromagnetism. Distractors are common mistakes.")
MODES = ["choice", "noul", "score"]
DEFAULT_N = 180
DEFAULT_SEED = 7

TIERS = [
    "1 single-digit",
    "2 multi-digit +/-",
    "3 multiply/divide",
    "4 fractions/%",
    "5 powers/roots/order",
    "6 algebra",
    "7 complex numbers",
    "8 complex analysis",
    "9 calculus",
    "10 linear alg./number th.",
    "11 physics: mechanics",
    "12 physics: E&M/waves",
]
GROUPS = {"arithmetic": TIERS[:5], "algebra & complex": TIERS[5:8], "advanced math": TIERS[8:10],
          "physics": TIERS[10:]}
MAG_BINS = ["0 to 9", "10 to 99", "100 to 999", "1,000 to 9,999", "10,000 or more"]
G = 9.8  # m/s²


def mag_bin(m):
    return sum(abs(m) >= e for e in (10, 100, 1000, 10000))


# ---------------------------------------------------------------- question generators
# Each returns (question, correct answer str, [distractor strs], |answer| as a float).

def num(x):
    """Exact if it has at most 2 decimals (112.5 stays 112.5), else 3 significant figures."""
    r = round(x, 2) if abs(x - round(x, 2)) < 1e-9 else float(f"{x:.3g}")
    return str(int(r)) if r == int(r) else f"{r:.10g}"


def numeric(rng, q, val, wrong, unit="", fmt=num, mag=None):
    """Build an item from a numeric answer and a list of plausible wrong values (common mistakes)."""
    ans = fmt(val) + unit
    ds = []
    for w in wrong:
        s = fmt(w) + unit
        if s != ans and s not in ds:
            ds.append(s)
    k = 1
    while len(ds) < 3:  # top up with nearby values if the mistakes collapsed onto each other
        s = fmt(val * (1 + 0.15 * k * rng.choice([-1, 1])) + rng.choice([-1, 1]) * k) + unit
        if s != ans and s not in ds:
            ds.append(s)
        k += 1
    return q, ans, ds[:3], abs(val) if mag is None else mag


def _near(rng, ans, spread):
    return [ans + rng.choice([-1, 1]) * rng.randint(1, spread) for _ in range(6)]


def single_digit(rng):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    op = rng.choice("+-")
    ans = a + b if op == "+" else a - b
    return numeric(rng, f"What is {a} {op} {b}?", ans, _near(rng, ans, 3))


def multi_digit(rng):
    a, b = rng.randint(100, 9999), rng.randint(100, 9999)
    op = rng.choice("+-")
    ans = a + b if op == "+" else a - b
    return numeric(rng, f"What is {a} {op} {b}?", ans, _near(rng, ans, 120), fmt=str)


def mul_div(rng):
    a, b = rng.randint(12, 99), rng.randint(3, 49)
    if rng.random() < 0.5:
        return numeric(rng, f"What is {a} × {b}?", a * b, _near(rng, a * b, a), fmt=str)
    return numeric(rng, f"What is {a * b} ÷ {b}?", a, _near(rng, a, 6), fmt=str)


def fractions_pct(rng):
    kind = rng.choice(["frac", "pct", "dec"])
    if kind == "frac":
        a, b = Fraction(rng.randint(1, 5), rng.randint(2, 7)), Fraction(rng.randint(1, 5), rng.randint(2, 7))
        ans = a + b
        wrong = [ans + Fraction(1, ans.denominator), Fraction(a.numerator + b.numerator, a.denominator + b.denominator),
                 ans * 2, ans + Fraction(1, ans.denominator + 1), ans + Fraction(2, ans.denominator + 3)]
        return numeric(rng, f"What is {a} + {b}?", ans, wrong, fmt=str, mag=float(ans))
    if kind == "pct":
        p, n = rng.choice([5, 10, 15, 20, 25, 40, 75]), rng.randint(2, 40) * 10
        ans = p * n / 100
        return numeric(rng, f"What is {p}% of {n}?", ans, [ans * 2, ans / 2, ans * 1.5, n * p / 10])
    a, b = round(rng.uniform(0.1, 9.9), 1), round(rng.uniform(0.1, 9.9), 1)
    ans = round(a * b, 2)
    return numeric(rng, f"What is {a} × {b}?", ans, [ans - 1.1, ans + 0.9, ans * 10], fmt=lambda x: f"{round(x, 2):g}")


def powers_roots(rng):
    kind = rng.choice(["pow", "root", "order"])
    if kind == "pow":
        b, e = rng.randint(2, 9), rng.randint(2, 4)
        return numeric(rng, f"What is {b}^{e}?", b ** e, [b * e, b ** (e - 1), b ** e + b], fmt=str)
    if kind == "root":
        r = rng.randint(4, 30)
        return numeric(rng, f"What is the square root of {r * r}?", r, _near(rng, r, 4), fmt=str)
    a, b, c = rng.randint(2, 9), rng.randint(2, 9), rng.randint(2, 9)
    ans = a + b * c
    return numeric(rng, f"What is {a} + {b} × {c}?", ans, [(a + b) * c, ans + 1, ans - 2], fmt=str)


def algebra(rng):
    if rng.random() < 0.5:
        x, a, b = rng.randint(-9, 12), rng.randint(2, 9), rng.randint(-20, 20)
        bs = f"+ {b}" if b >= 0 else f"- {-b}"
        return numeric(rng, f"Solve for x: {a}x {bs} = {a * x + b}", x, _near(rng, x, 4),
                       fmt=lambda v: f"x = {v}")
    r1, r2 = rng.sample(range(1, 10), 2)
    lo, hi = min(r1, r2), max(r1, r2)
    ds = [f"x = {-hi} or x = {-lo}", f"x = {lo} or x = {hi + 1}",
          f"x = 1 or x = {lo * hi}" if lo != 1 else f"x = 2 or x = {lo * hi}"]
    return f"Solve for x: x² - {lo + hi}x + {lo * hi} = 0", f"x = {lo} or x = {hi}", ds, hi


def _cstr(z):
    a, b = int(round(z.real)), int(round(z.imag))
    if b == 0:
        return str(a)
    ib = "i" if abs(b) == 1 else f"{abs(b)}i"
    if a == 0:
        return ("-" if b < 0 else "") + ib
    return f"{a} {'+' if b > 0 else '-'} {ib}"


def complex_item(q, ans, wrong):
    a = _cstr(ans)
    ds = list(dict.fromkeys(_cstr(w) for w in wrong if _cstr(w) != a))[:3]
    return q, a, ds, abs(ans)


def complex_nums(rng):
    kind = rng.choice(["add", "mul", "ipow", "mod", "conj"])
    z1 = complex(rng.randint(-6, 6) or 1, rng.randint(-6, 6) or 2)
    z2 = complex(rng.randint(-6, 6) or 3, rng.randint(-6, 6) or -1)
    if kind == "add":
        s = z1 + z2
        return complex_item(f"What is ({_cstr(z1)}) + ({_cstr(z2)})?", s, [s.conjugate(), s + 1j, z1 - z2, s + 1, s - 2])
    if kind == "mul":
        p = z1 * z2
        naive = complex(z1.real * z2.real + z1.imag * z2.imag, z1.real * z2.imag + z1.imag * z2.real)  # i² = +1
        return complex_item(f"What is ({_cstr(z1)}) × ({_cstr(z2)})?", p,
                            [naive, complex(z1.real * z2.real, z1.imag * z2.imag), p.conjugate(), p + 2, p - 1j])
    if kind == "ipow":
        n = rng.randint(2, 40)
        cyc = [1, 1j, -1, -1j]
        return complex_item(f"What is i^{n}?", cyc[n % 4], [c for c in cyc if c != cyc[n % 4]])
    if kind == "mod":
        a, b = rng.choice([(3, 4), (5, 12), (6, 8), (8, 15), (9, 12), (7, 24)])
        z = complex(a * rng.choice([-1, 1]), b * rng.choice([-1, 1]))
        m = int(abs(z))
        return numeric(rng, f"What is the modulus |{_cstr(z)}|?", m, [a + b, a * a + b * b, m + 1], fmt=str)
    c = z1.conjugate()
    return complex_item(f"What is the complex conjugate of {_cstr(z1)}?", c,
                        [-z1, -c, complex(z1.imag, z1.real), z1 + 2])


def complex_analysis(rng):
    kind = rng.choice(["div", "demoivre", "arg", "euler", "prodmod"])
    if kind == "div":
        w = complex(rng.randint(-5, 5) or 2, rng.randint(-5, 5) or 1)
        z2 = complex(rng.randint(1, 4), rng.choice([-1, 1]) * rng.randint(1, 4))
        z1 = w * z2
        return complex_item(f"What is ({_cstr(z1)}) ÷ ({_cstr(z2)})?", w,
                            [w.conjugate(), z1 * z2.conjugate(), -w, w + 1j, w * 1j])
    if kind == "demoivre":
        n = rng.randint(3, 10)
        z = complex(1, 1) ** n
        z = complex(round(z.real), round(z.imag))
        return complex_item(f"Using De Moivre's theorem, what is (1 + i)^{n}?", z,
                            [complex(2 ** n, 0), z.conjugate(), -z, (1 + 1j) ** (n - 1), z * 1j])
    if kind == "arg":
        z, deg = rng.choice([("1 + i", 45), ("i", 90), ("-1 + i", 135), ("-1", 180), ("1 - i", -45), ("-i", -90),
                             ("√3 + i", 30), ("1 + √3i", 60), ("-√3 + i", 150)])
        wrong = [deg + 90, 90 - deg, -deg, deg + 45, 180 - deg]
        return numeric(rng, f"What is the principal argument of {z}, in degrees?", deg, wrong, unit="°", fmt=str,
                       mag=abs(deg))
    if kind == "euler":
        k = rng.randint(1, 12)
        cyc = [1, 1j, -1, -1j]
        return complex_item(f"What is e^(iπ·{k}/2)?", cyc[k % 4], [c for c in cyc if c != cyc[k % 4]])
    (a, b), (c, d) = rng.sample([(3, 4), (5, 12), (6, 8), (8, 15), (7, 24)], 2)
    z1, z2 = complex(a, b), complex(c, rng.choice([-1, 1]) * d)
    m = int(round(abs(z1) * abs(z2)))
    return numeric(rng, f"What is |({_cstr(z1)}) × ({_cstr(z2)})|?", m,
                   [int(abs(z1) + abs(z2)), a * c + b * d, m + int(abs(z1)), m * 2], fmt=str)


def calculus(rng):
    kind = rng.choice(["deriv", "integral", "limit_poly", "limit_sin", "integral_sin"])
    if kind == "deriv":
        a, b, c, d, x = rng.randint(1, 5), rng.randint(-6, 6), rng.randint(-9, 9), rng.randint(-9, 9), rng.randint(-4, 4) or 2
        f = f"{a}x³ {'+' if b >= 0 else '-'} {abs(b)}x² {'+' if c >= 0 else '-'} {abs(c)}x {'+' if d >= 0 else '-'} {abs(d)}"
        ans = 3 * a * x * x + 2 * b * x + c
        wrong = [a * x ** 3 + b * x ** 2 + c * x + d, a * x * x + b * x + c, 3 * a * x * x + b * x + c, ans + 2 * x]
        return numeric(rng, f"If f(x) = {f}, what is f′({x})?", ans, wrong, fmt=str)
    if kind == "integral":
        m, n, k = rng.randint(1, 4), rng.randint(1, 5), rng.randint(1, 5)
        a, b = 3 * m, 2 * n
        ans = m * k ** 3 + n * k ** 2
        wrong = [a * k ** 3 + b * k ** 2, a * k ** 2 + b * k, 2 * a * k + b, ans + k]
        return numeric(rng, f"What is the definite integral of {a}x² + {b}x from x = 0 to x = {k}?", ans, wrong, fmt=str)
    if kind == "limit_poly":
        a = rng.randint(2, 12)
        return numeric(rng, f"What is the limit of (x² - {a * a}) / (x - {a}) as x approaches {a}?", 2 * a,
                       [0, a, a * a, a * a * 2], fmt=str)
    if kind == "limit_sin":
        k = rng.randint(2, 9)
        return f"What is the limit of sin({k}x) / x as x approaches 0?", str(k), ["0", "1", f"1/{k}"], k
    k = rng.randint(2, 9)
    return numeric(rng, f"What is the integral of {k}·sin(x) from x = 0 to x = π?", 2 * k, [0, k, -2 * k, k * 3.14],
                   fmt=num)


def linalg_numtheory(rng):
    kind = rng.choice(["det2", "det3", "gcd", "modpow", "trace"])
    if kind == "det2":
        a, b, c, d = (rng.randint(-9, 9) for _ in range(4))
        return numeric(rng, f"What is the determinant of the matrix [[{a}, {b}], [{c}, {d}]]?", a * d - b * c,
                       [a * d + b * c, b * c - a * d, a * b - c * d, a + d], fmt=str)
    if kind == "det3":
        m = [[rng.randint(-4, 4) for _ in range(3)] for _ in range(3)]
        det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        no_sign = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) + m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
                   + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))  # forgot the alternating sign
        diag = m[0][0] * m[1][1] * m[2][2]
        return numeric(rng, f"What is the determinant of the matrix {m}?", det, [no_sign, -det, diag, det + 6], fmt=str)
    if kind == "gcd":
        g = rng.randint(2, 30)
        p, q = rng.choice([(2, 3), (3, 4), (3, 5), (4, 7), (5, 6), (5, 8), (7, 9)])
        return numeric(rng, f"What is the greatest common divisor of {g * p} and {g * q}?", g,
                       [g * 2, g * p, g * q, g + 1], fmt=str)
    if kind == "modpow":
        a, b, m = rng.randint(2, 9), rng.randint(5, 30), rng.randint(7, 23)
        ans = pow(a, b, m)
        return numeric(rng, f"What is {a}^{b} mod {m}?", ans,
                       [(a * b) % m, (ans + 1) % m, (ans + 2) % m, (ans + m - 1) % m], fmt=str)
    a, b, c, d = (rng.randint(-9, 9) for _ in range(4))
    return numeric(rng, f"What is the sum of the eigenvalues of the matrix [[{a}, {b}], [{c}, {d}]]?", a + d,
                   [a * d - b * c, a + b + c + d, a * d, a - d], fmt=str)


def physics_mech(rng):
    kind = rng.choice(["vel", "dist", "fall", "ke", "pe", "inelastic", "height"])
    if kind == "vel":
        u, a, t = rng.randint(0, 20), rng.randint(1, 6), rng.randint(2, 12)
        return numeric(rng, f"A car moving at {u} m/s accelerates at {a} m/s² for {t} s. What is its final speed?",
                       u + a * t, [u * a * t, u + a * t * t, a * t, u + a * t / 2], " m/s")
    if kind == "dist":
        u, a, t = rng.randint(0, 15), rng.randint(1, 6), rng.randint(2, 10)
        return numeric(rng, f"A cart starts at {u} m/s and accelerates at {a} m/s² for {t} s. How far does it travel?",
                       u * t + a * t * t / 2, [u * t + a * t * t, a * t * t / 2, u * t, (u + a * t) * t], " m")
    if kind == "fall":
        h = rng.choice([5, 10, 20, 45, 80, 125, 180])
        return numeric(rng, f"An object is dropped from rest from a height of {h} m. How long does it take to hit the "
                       f"ground? (g = 9.8 m/s²)", (2 * h / G) ** 0.5, [(h / G) ** 0.5, 2 * h / G, h / G], " s")
    if kind == "ke":
        m, v = rng.randint(1, 50), rng.randint(2, 30)
        return numeric(rng, f"What is the kinetic energy of a {m} kg object moving at {v} m/s?", m * v * v / 2,
                       [m * v * v, m * v / 2, m * v, m * m * v / 2], " J")
    if kind == "pe":
        m, h = rng.randint(1, 40), rng.randint(2, 50)
        return numeric(rng, f"What is the gravitational potential energy of a {m} kg mass raised {h} m? (g = 9.8 m/s²)",
                       m * G * h, [m * h, m * G * h / 2, m * G, m * G * h * h], " J")
    if kind == "inelastic":
        m1, m2, v1 = rng.randint(1, 10), rng.randint(1, 10), rng.randint(2, 20)
        return numeric(rng, f"A {m1} kg cart moving at {v1} m/s hits a {m2} kg cart at rest and they stick together. "
                       f"What is their speed after the collision?", m1 * v1 / (m1 + m2),
                       [m1 * v1 / m2 if m2 != m1 + m2 else v1 / 2, v1, (m1 + m2) * v1 / m1, v1 / 2], " m/s")
    v = rng.randint(5, 40)
    return numeric(rng, f"A ball is thrown straight up at {v} m/s. What maximum height does it reach? (g = 9.8 m/s²)",
                   v * v / (2 * G), [v * v / G, v / (2 * G), v / G, v * v / 2], " m")


def physics_em(rng):
    kind = rng.choice(["ohm", "power", "parallel", "wave", "cap", "coulomb"])
    if kind == "ohm":
        v, r = rng.choice([6, 9, 12, 24, 120, 230]), rng.choice([2, 3, 4, 5, 8, 10, 20, 50])
        return numeric(rng, f"A {v} V battery is connected across a {r} Ω resistor. What current flows?", v / r,
                       [v * r, r / v, v / r / 2], " A")
    if kind == "power":
        v, r = rng.choice([6, 9, 12, 24, 120]), rng.choice([2, 3, 4, 6, 10, 12])
        return numeric(rng, f"What power is dissipated by a {r} Ω resistor with {v} V across it?", v * v / r,
                       [v / r, v * r, v * v * r, v * v / (2 * r)], " W")
    if kind == "parallel":
        r1, r2 = rng.sample([2, 3, 4, 6, 8, 10, 12, 15, 20, 30], 2)
        return numeric(rng, f"What is the equivalent resistance of a {r1} Ω and a {r2} Ω resistor in parallel?",
                       r1 * r2 / (r1 + r2), [r1 + r2, (r1 + r2) / 2, r1 * r2, abs(r1 - r2)], " Ω")
    if kind == "wave":
        f, lam = rng.choice([50, 100, 256, 440, 1000]), rng.choice([0.5, 0.75, 1.2, 2, 3.4])
        return numeric(rng, f"A wave has frequency {f} Hz and wavelength {lam} m. What is its speed?", f * lam,
                       [f / lam, lam / f, f + lam], " m/s")
    if kind == "cap":
        c, v = rng.choice([10, 47, 100, 220, 470]), rng.choice([5, 9, 12, 25, 50])
        return numeric(rng, f"How much energy, in millijoules, is stored in a {c} μF capacitor charged to {v} V?",
                       c * v * v / 2000, [c * v * v / 1000, c * v / 2000, c * v * v / 2], " mJ")
    q1, q2, r = rng.randint(1, 9), rng.randint(1, 9), rng.choice([0.1, 0.2, 0.3, 0.5, 1.0])
    k = 8.99e9
    f = k * q1 * q2 * 1e-12 / r ** 2
    return numeric(rng, f"What is the electric force between charges of {q1} μC and {q2} μC placed {r} m apart? "
                   f"(k = 8.99 × 10⁹ N·m²/C²)", f, [k * q1 * q2 * 1e-12 / r, f * 2, f / 10], " N")


GENERATORS = [single_digit, multi_digit, mul_div, fractions_pct, powers_roots, algebra, complex_nums,
              complex_analysis, calculus, linalg_numtheory, physics_mech, physics_em]


CATEGORIES = TIERS
SCORE_LABELS = MAG_BINS


def build(n=DEFAULT_N, seed=DEFAULT_SEED):
    """The items, deterministic in (n, seed). Generators cycle, so n=180 gives 15 per tier."""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        tier = i % len(GENERATORS)
        q, ans, ds, mag = GENERATORS[tier](rng)
        ds = list(dict.fromkeys(d for d in ds if d != ans))
        assert len(ds) == 3, (q, ans, ds)
        opts = [ans] + ds
        rng.shuffle(opts)
        shown_true = rng.random() < 0.5
        items.append(dict(id=i, category=TIERS[tier], prompt=q, answer=ans, options=opts,
                          shown_answer=ans if shown_true else rng.choice(ds), shown_is_correct=shown_true,
                          magnitude=mag, score_level=mag_bin(mag)))
    return items


def state(item):
    return {"problem": item["prompt"]}


def instructions(item):
    """The instruction text for each mode. The runner adds the options (choice) and SCORE_LABELS (score)."""
    q = item["prompt"]
    return {
        "choice": f"{q} Pick the correct answer.",
        "noul": f"Is this answer correct? {q} Answer: {item['shown_answer']}",
        "score": f"How large is the magnitude (absolute value) of the answer to: {q}",
    }
