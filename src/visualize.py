import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_score_distributions(columns:list[pd.Series], title:str="Scoring distribution"):    
    # Filter columns that actually exist
    
    plt.figure(figsize=(12, 7))
    
    # tab10 provides exactly 10 distinct colors
    palette = sns.color_palette("tab10")
    
    for i, col in enumerate(columns):
        color = palette[i % 10]  # Repeat colors if > 10
        sns.histplot(
            data=col.dropna(), 
            binwidth=1,       # Gom nhóm theo khoảng là 1 (VD: từ 5.0 đến cận 6.0)
            stat='count',     # Đảm bảo trục Y hiển thị số lượng đếm gốc
            color=color, 
            label=col.name, 
            alpha=0.4,        # Độ trong suốt để dễ nhìn khi các cột đè lên nhau
            element='bars',   # Hiển thị dạng cột truyền thống
            kde=False         # Tắt đường viền nội suy (nếu chỉ muốn xem số liệu thô)
        )
        
    plt.title(title, fontsize=16)
    plt.xlabel('Score', fontsize=14)
    plt.ylabel('Articles Counting', fontsize=14)
    plt.legend(title='Score Set')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()