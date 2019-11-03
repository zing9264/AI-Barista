# 中正大學--AI Barista 108/11/2

**Purpose**
---------------------------------------------
> 訓練出一名沖煮動作穩定的咖啡師，往往需要數年的經驗累積，而機器手臂可以避免掉不穩定的人為因素，獲得品質一致的手沖咖啡，若能有完美的沖煮系統，便能為業界帶來極大的效益。目前市面上有許多科技公司推出的機械手臂沖泡系統，卻都有一個致命的問題點—「機器無法觀察咖啡的研磨、烘焙程度進行調整，只能用固定的手法沖煮」。我們認為，這樣的沖泡方式不過是較昂貴的咖啡機。因此，我們打算對此進行優化，開發一套智慧咖啡沖煮系統。

> 在本專題中，不僅實現了用機器手臂手沖咖啡，更利用了K-mean、CLAHE演算法、機器學習等技術，建構出一套CNN的回歸模型，使電腦能夠根據咖啡粉圖片，快速分析出粉末當下的粗細重量分布，並根據此研磨分布建構出多條合適的沖煮曲線，即使在未知研磨度的狀況下，也能為機械手臂注入靈魂，化身成一名專業咖啡手沖師，完美萃取咖啡中的曼妙滋味。

> 透過本系統，使用者只須拍攝一張磨好的咖啡粉照片，便能為您送上一杯最適合該研磨度的美味咖啡。

National Chung Cheng University, CS 
2019 Independent Study

**Methods**
---------------------------------------------
>1. Grind the beans with the coffee grinder.
>2. Take a picture for coffee powder.
>3. Use image recognition to analysis composition of powder (think or thin).
>4. Calculate optimal pouring water curve by variables from above step.
>5. Control robotic arm to brew pour-over-coffee with curve.

**Members**
---------------------------------------------
### 高靖智 毛胤年 楊淨 鄭傑予
### 國立中正大學 資訊工程學系
###### National Chung Cheng University, Taiwan.
###### Computer Science and Information Engineering.
