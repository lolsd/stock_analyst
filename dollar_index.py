import yfinance as yf
import matplotlib.pyplot as plt

# 1. 获取美元指数日线数据（可改period，如'1y'、'5y'、'max'）
dxy = yf.Ticker("DX-Y.NYB")
df = dxy.history(period="1y")  # 最近1年日线

# 只保留日期和收盘价
df = df[["Close"]].copy()
df.reset_index(inplace=True)
df["Date"] = df["Date"].dt.date

# 2. 打印最新5天数据（核对用）
print("最近5天美元指数：")
print(df.tail())

# 3. 绘制折线图
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Close"], color="#1f77b4", linewidth=2)

plt.title("美元指数（DXY）每日收盘价", fontsize=15)
plt.xlabel("日期")
plt.ylabel("指数")
plt.grid(True, linestyle="--", alpha=0.6)

# 日期标签不重叠
plt.xticks(rotation=45)
plt.tight_layout()

# 保存图片 + 显示
plt.savefig("dxy_daily.png", dpi=300)
plt.show()