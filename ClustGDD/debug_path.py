import os

# 1. 看看 Python 现在站在哪里
current_dir = os.getcwd()
print(f"我现在站的位置是: {current_dir}")

# 2. 看看 Python 试图找的路径存不存在
target_file = os.path.join(current_dir, 'data', 'flickr', 'adj_full.npz')
print(f"我正在尝试找这个文件: {target_file}")

if os.path.exists(target_file):
    print("✅ 成功！文件找到了！")
else:
    print("❌ 失败！文件不在那里。")
    
    # 3. 看看 data/flickr 里面到底有什么
    flickr_dir = os.path.join(current_dir, 'data', 'flickr')
    if os.path.exists(flickr_dir):
        print(f"文件夹 {flickr_dir} 存在，里面的东西是：")
        print(os.listdir(flickr_dir))
    else:
        print(f"❌ 连文件夹 {flickr_dir} 都找不到！请检查 'data' 或 'flickr' 名字是不是写错了（注意大小写）。")