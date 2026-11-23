import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy.integrate import solve_ivp

torch.manual_seed(42)
np.random.seed(42)

# 1. 생리학적 파라미터 및 Azithromycin 기본 정보
PHYS = {
    "V_p": 3.0, "V_lung": 0.5, "V_rest": 40.0, "V_csf": 0.15,
    "Q_lung": 300.0, "Q_rest": 250.0, "Q_csf": 0.02,
}

AZITHROMYCIN_SMILES = (
    "CCC1C(C(C(N(CC(CC(C(C(C(C(C(=O)O1)C)OC2CC(C(C(O2)C)O)(C)OC)C)"
    "OC3C(C(CC(O3)C)N(C)C)O)(C)O)C)C)C)O)(C)O"
)
DOSE = 500.0      # mg (단회 1시간 IV Infusion)
T_INF = 1.0       # h
T_MAX = 24.0      # h (관측 기간)
CL = 38.0         # L/h
FU = 0.70         # 혈장 비결합 분율
KP_REST = 50.0    # 기타 조직 분배계수 (고정 가정)

# 2. RDKit으로 2D 물성 기반 초기 추정치(Prior) 계산 (리소좀 축적/수막염 미반영)
mol = Chem.MolFromSmiles(AZITHROMYCIN_SMILES)
mw = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
tpsa = Descriptors.TPSA(mol)

P_val = 10 ** np.clip(logp, -3.0, 4.0)
kp_lung_prior = 0.35 + 0.05 * P_val
log_bb = 0.152 * logp - 0.0148 * tpsa + 0.139
ps_bbb_prior = 0.05 * (10 ** np.clip(log_bb, -4.0, 1.0)) * ((350.0 / max(mw, 350.0)) ** 1.5)

print(f"[RDKit 2D Prior] Kp_lung: {kp_lung_prior:.3f} | PS_bbb: {ps_bbb_prior:.5f}")

# 3. 가상 임상 데이터 생성 (숨겨진 실제 값: 리소좀 축적 Kp_lung=15.0, 수막염 PS_bbb=0.015)
TRUE_KP_LUNG = 15.0
TRUE_PS_BBB = 0.015

def smooth_infusion_np(t):
    # t=1.0h에서 부드럽게 0으로 떨어지는 1시간 점적주사 근사
    return (DOSE / T_INF) * (1.0 / (1.0 + np.exp(15.0 * (t - T_INF))))

def true_ode(t, y):
    C_p, C_lung, C_rest, C_csf = y
    R_inf = smooth_infusion_np(t)
    flux_lung = PHYS["Q_lung"] * (C_p - C_lung / TRUE_KP_LUNG)
    flux_rest = PHYS["Q_rest"] * (C_p - C_rest / KP_REST)
    flux_csf = TRUE_PS_BBB * (FU * C_p - C_csf)

    return [
        (R_inf - CL * C_p - flux_lung - flux_rest - flux_csf) / PHYS["V_p"],
        flux_lung / PHYS["V_lung"],
        flux_rest / PHYS["V_rest"],
        (flux_csf - PHYS["Q_csf"] * C_csf) / PHYS["V_csf"],
    ]

# 참값 궤적 및 소수(8개 시점)의 노이즈 포함 임상 관측 데이터 생성
t_obs_np = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 18.0, 24.0])
sol_true = solve_ivp(true_ode, (0, T_MAX), [0.0, 0.0, 0.0, 0.0], method="Radau", t_eval=t_obs_np, max_step=0.05)

# 실제 임상에서는 혈장(0), 폐/ELF(1), CSF(3)만 채취 가능 (C_rest는 미관측)
noise_level = 0.05  # 5% 상대 오차 노이즈
obs_p = sol_true.y[0] * (1.0 + noise_level * np.random.randn(len(t_obs_np)))
obs_lung = sol_true.y[1] * (1.0 + noise_level * np.random.randn(len(t_obs_np)))
obs_csf = sol_true.y[3] * (1.0 + noise_level * np.random.randn(len(t_obs_np)))

# 구획별 스케일 정규화 상수 (각 구획 최대 농도 기준)
SCALE_P = float(np.max(obs_p))
SCALE_LUNG = float(np.max(obs_lung))
SCALE_REST = float(np.max(sol_true.y[2]))  # 물리 잔차 스케일링용
SCALE_CSF = float(np.max(obs_csf))

t_obs_t = torch.tensor(t_obs_np / T_MAX, dtype=torch.float32).view(-1, 1)
obs_p_t = torch.tensor(obs_p, dtype=torch.float32).view(-1, 1)
obs_lung_t = torch.tensor(obs_lung, dtype=torch.float32).view(-1, 1)
obs_csf_t = torch.tensor(obs_csf, dtype=torch.float32).view(-1, 1)

# 4. Minimal PBPK-PINN 모델 정의
class PBPK_PINN(nn.Module):
    def __init__(self, init_kp_lung, init_ps_bbb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 4)  # 출력: [C_p, C_lung, C_rest, C_csf] (정규화된 크기)
        )
        # RDKit Prior에서 출발하는 로그 공간 학습 파라미터 (양수 제약 보장)
        self.log_kp_lung = nn.Parameter(torch.tensor([np.log(init_kp_lung)], dtype=torch.float32))
        self.log_ps_bbb = nn.Parameter(torch.tensor([np.log(init_ps_bbb)], dtype=torch.float32))

    def forward(self, t_norm):
        # Hard Initial Condition: t=0일 때 모든 농도 0 강제 + Softplus로 음수 농도 방지
        raw = F.softplus(self.net(t_norm)) * t_norm
        scales = torch.tensor([[SCALE_P, SCALE_LUNG, SCALE_REST, SCALE_CSF]], dtype=torch.float32)
        return raw * scales

model = PBPK_PINN(kp_lung_prior, ps_bbb_prior)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1500, gamma=0.5)

# 초반 점적주사 구간(0~2h)에 밀집된 비균일 콜로케이션 포인트 생성
t_colloc_np = np.unique(np.concatenate([
    np.linspace(0.01, 2.0, 60),
    np.linspace(2.0, T_MAX, 60)
]))
t_colloc_norm = torch.tensor(t_colloc_np / T_MAX, dtype=torch.float32).view(-1, 1).requires_grad_(True)

# 5. 학습 루프
hist_kp, hist_ps = [], []

for epoch in range(4501):
    optimizer.zero_grad()

    # [손실 1] 데이터 손실 (정규화된 상대 오차)
    pred_obs = model(t_obs_t)
    loss_data = (
        torch.mean(((pred_obs[:, 0:1] - obs_p_t) / SCALE_P) ** 2)
        + torch.mean(((pred_obs[:, 1:2] - obs_lung_t) / SCALE_LUNG) ** 2)
        + torch.mean(((pred_obs[:, 3:4] - obs_csf_t) / SCALE_CSF) ** 2)
    )

    # [손실 2] 물리 손실 (4개 구획 미분방정식 잔차)
    y_col = model(t_colloc_norm)
    C_p, C_lung, C_rest, C_csf = y_col[:, 0:1], y_col[:, 1:2], y_col[:, 2:3], y_col[:, 3:4]

    # 연쇄 법칙: dC/dt = (dC/dt_norm) / T_MAX
    dC_dt = []
    for i in range(4):
        grad_i = torch.autograd.grad(
            y_col[:, i:i+1], t_colloc_norm,
            grad_outputs=torch.ones_like(y_col[:, i:i+1]),
            create_graph=True
        )[0] / T_MAX
        dC_dt.append(grad_i)

    kp_lung = torch.exp(model.log_kp_lung)
    ps_bbb = torch.exp(model.log_ps_bbb)

    t_real = t_colloc_norm * T_MAX
    R_inf = (DOSE / T_INF) * torch.sigmoid(-15.0 * (t_real - T_INF))

    flux_lung = PHYS["Q_lung"] * (C_p - C_lung / kp_lung)
    flux_rest = PHYS["Q_rest"] * (C_p - C_rest / KP_REST)
    flux_csf = ps_bbb * (FU * C_p - C_csf)

    ode_p = (R_inf - CL * C_p - flux_lung - flux_rest - flux_csf) / PHYS["V_p"]
    ode_lung = flux_lung / PHYS["V_lung"]
    ode_rest = flux_rest / PHYS["V_rest"]
    ode_csf = (flux_csf - PHYS["Q_csf"] * C_csf) / PHYS["V_csf"]

    # 각 ODE 잔차를 해당 구획 스케일로 정규화하여 Stiffness 불균형 해소
    loss_phys = (
        torch.mean(((dC_dt[0] - ode_p) / (SCALE_P * 10.0)) ** 2)
        + torch.mean(((dC_dt[1] - ode_lung) / (SCALE_LUNG * 10.0)) ** 2)
        + torch.mean(((dC_dt[2] - ode_rest) / SCALE_REST) ** 2)
        + torch.mean(((dC_dt[3] - ode_csf) / SCALE_CSF) ** 2)
    )

    loss = loss_data + 0.5 * loss_phys
    loss.backward()
    optimizer.step()
    scheduler.step()

    hist_kp.append(kp_lung.item())
    hist_ps.append(ps_bbb.item())

    if epoch % 900 == 0:
        print(
            f"Epoch {epoch:4d} | Loss: {loss.item():.5f} | "
            f"Kp_lung: {kp_lung.item():.2f} (True: {TRUE_KP_LUNG}) | "
            f"PS_bbb: {ps_bbb.item():.5f} (True: {TRUE_PS_BBB})"
        )

# 6. 결과 시각화
t_eval_np = np.linspace(0, T_MAX, 300)
sol_eval = solve_ivp(true_ode, (0, T_MAX), [0.0, 0.0, 0.0, 0.0], method="Radau", t_eval=t_eval_np, max_step=0.05)

model.eval()
with torch.no_grad():
    pred_eval = model(torch.tensor(t_eval_np / T_MAX, dtype=torch.float32).view(-1, 1)).numpy()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# [그래프 1] 폐(ELF) 및 혈장 농도 피팅 결과
axes[0].plot(t_eval_np, sol_eval.y[1], "g--", label="True Lung/ELF")
axes[0].plot(t_eval_np, pred_eval[:, 1], "g-", linewidth=2, label="PINN Lung/ELF")
axes[0].scatter(t_obs_np, obs_lung, color="darkgreen", s=30, zorder=5)
axes[0].plot(t_eval_np, sol_eval.y[0], "b--", label="True Plasma")
axes[0].plot(t_eval_np, pred_eval[:, 0], "b-", linewidth=2, label="PINN Plasma")
axes[0].scatter(t_obs_np, obs_p, color="navy", s=30, zorder=5)
axes[0].set_title("Plasma & Lung/ELF Concentration")
axes[0].set_xlabel("Time (h)")
axes[0].set_ylabel("Concentration (mg/L)")
axes[0].legend(fontsize=8.5)
axes[0].grid(True, alpha=0.3)

# [그래프 2] 뇌척수액(CSF) 농도 피팅 결과
axes[1].plot(t_eval_np, sol_eval.y[3], "r--", label="True CSF (Meningitis)")
axes[1].plot(t_eval_np, pred_eval[:, 3], "r-", linewidth=2, label="PINN CSF")
axes[1].scatter(t_obs_np, obs_csf, color="darkred", s=30, zorder=5)
axes[1].set_title("CSF Concentration (Meningitis)")
axes[1].set_xlabel("Time (h)")
axes[1].set_ylabel("Concentration (mg/L)")
axes[1].legend(fontsize=8.5)
axes[1].grid(True, alpha=0.3)

# [그래프 3] RDKit Prior에서 참값으로 수렴하는 파라미터 궤적
ax_kp = axes[2]
ax_ps = ax_kp.twinx()
l1 = ax_kp.plot(hist_kp, "g-", linewidth=2, label="Est. Kp_lung")
l2 = ax_kp.axhline(TRUE_KP_LUNG, color="g", linestyle="--", alpha=0.7, label="True Kp_lung")
l3 = ax_ps.plot(hist_ps, "r-", linewidth=2, label="Est. PS_bbb")
l4 = ax_ps.axhline(TRUE_PS_BBB, color="r", linestyle="--", alpha=0.7, label="True PS_bbb")
ax_kp.set_title("Inverse Identification (from RDKit Prior)")
ax_kp.set_xlabel("Epoch")
ax_kp.set_ylabel("Kp_lung", color="g")
ax_ps.set_ylabel("PS_bbb (L/h)", color="r")
lines = l1 + [l2] + l3 + [l4]
ax_kp.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
ax_kp.grid(True, alpha=0.3)

plt.tight_layout()
# plt.savefig("rdkit_pinn_inverse_azithromycin.png", dpi=150)
plt.show()