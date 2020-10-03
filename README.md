# Awesome
 **算法秒撕**

# ----排序---- #

## 快速排序 *
     参考算法导论
     partition,将数组in-place地分成比pivot小的一边和比pivot大的一边；
     多少有些分治的意思
     随机产生pivot，并将其换到最后
     partition结束后，将其换到i+1的位置 

## 堆排序 *
     参考算法导论
     最大堆  A.val < A.parent.val
     整个二叉树除了最底一层都是满的
     i  LEFT => 2i; RIGHT => 2i+1
     保持堆的顺序正确 O(lgn)
     由无序数组建堆 O(n)
     堆排序  O(nlgn)  => 抽取顶部元素后，用最末元素补上并调整树

## 冒泡排序 

# ----分治---- #

## 第k大的数 *
    partition，而且单边就行，O(n)
## 数组连续的最大和 *
    和一些炒股问题类似
    数组中有正有负有0
    解法1：
    分治 => max{左边最大，右边最大，跨中间最大}
    解法2（LeetCode & 剑指offer63）：
    股票低点买入，高点售出
    维护一个当前的最低值，和当前的最大利润；最后返回当前最大利润
    int max_ = INT_MIN;
    int min_ = INT_MAX;
    for(int v: prices){
        min_ = min(min_,v);
        max_ = max(max_,v-min_);
    }
    if (max_ <= 0) return 0;
    return max_;

# ----链表---- #
**链表题目大量涉及指针，注意代码严谨性和完整性，检查是否为空**

##链表判断是否有环 *
      LeetCode141, 亦可参考剑指offer
      快慢节点，有则相遇
   
##链表找环的入口 *
      Leetcode142
      剑指offer的解法很啰嗦
      实际上，相遇点到入口距离 +k个环长等于 起点到入口的距离
      因此快慢二点相遇后，将其中一个调到起点，再同步走，再次相遇处就是入口起点
      证明： 设起点到入口距离a, 入口点到一次相遇点为b，一次相遇点到入口为c。 2(a+b) = a + b + k(b+c) ; k>=1
      a = (k-1)(b+c) + c 


## 链表倒数第k个节点
    剑指offer22 。双指针，让第一个指针先走k-1步

## 合并两个排序链表
    剑指offer25
    用递归合并，注意有链表为空指针的情况。确定当前节点后，后序依旧是两个有序链表合并

## 两个链表的第一个公共节点
    剑指offer52 
    如果相交也只可能是Y形状.遍历两链表，找出长度差。让长的先走几步。
    像统计链表长度这种用了两次涉及较多指针，应当拆分成一个独立函数

## 链表反转 *
    LeetCode206 ,剑指offer24; 话不多说，直接代码吧
    ListNode* reverseList(ListNode* head) {
        ListNode *cur = head;
        ListNode *prev = NULL;
        while(cur!=NULL){
            ListNode *nextTmp = cur->next;
            cur->next = prev;
            prev = cur;
            cur = nextTmp;
        }
        return prev;
    }

## 链表部分反转

# ----动态规划---- #
## 最长公共子序列  *
    LeetCode 1143
    R[i][j] = max(R[i-1][j],R[i][j-1]) if S[i]!=T[j]
             = R[i-1][j-1] +1           else


## 最长回文子序列
    LeetCode516 对字符串求逆，然后求最长公共子序列

## 最长递增子序列 *
    LeetCode 300
    以i结尾的最长子序列
    R[i] ,充分利用已经计算的R[]值
    for i from 1 to n:
        from j from 1 to i:
             if nums[j] < nums[i]:
                 R[i] = max(R[i], R[j]+1)

## 最长公共子串 *
    R[i][j] = 0             if s[i] != t[j]
           = R[i-1][j-1]+1  else

## 最长回文子串 
    Leetcode5 对字符串求逆，然后求LCS即可

## 子数组最大乘积 
    Leetcode152 （数组可能含有0与负值）
    维护以i结尾的最大值与最小值imax,imin；以及整体的最大值
    A[i] < 0 => swap(imax, imin)
    imax = max(A[i], imax)
    imin = min(A[i], imin)

## 股票买卖(冷却一天，多次买卖)
    LeetCode309 必须先卖光后买
    状态比较多，用自动机去理解
    S0是休息状态，可以买入和休息
    S1是持股状态，可以休息和卖出
    S2是售出状态，只能休息
    s0[i] = max(s0[i - 1], s2[i - 1]); // Stay at s0, or rest from s2
    s1[i] = max(s1[i - 1], s0[i - 1] - prices[i]); // Stay at s1, or buy from s0
    s2[i] = s1[i - 1] + prices[i]; // Only one way from s1
    
## 股票买卖(有交易费)
    同样是理解成自动机的状态
    for (int i = 1; i<days; i++) {
            buy[i] = Math.max(buy[i - 1], sell[i - 1] - prices[i] - fee); // keep the same as day i-1, or buy from sell status at day i-1
            sell[i] = Math.max(sell[i - 1], buy[i - 1] + prices[i]); // keep the same as day i-1, or sell from buy status at day i-1
        }

## 目标和
      LeetCode494
      给定数组，允许用+和-,求有多少种达到目标和的方式  
      很容易联想到递归--有记忆递归--自底向上DP  dp[index][sum]

# ----栈\队列\堆-----

##两栈构成队列 
    剑指offer9  
    栈1正常压栈，要删除则弹到栈2（栈2非空时），栈2专门负责删队列头部

##两队列构成栈  
    剑指offer9 更加无聊。要出栈就将队列其他元素全部移到另一个队列中。每次总有一个队列为空

## 依据身高重构队列
    LeetCode406  数组(h,k) k指队伍前面身高>=h的人的数量，重构
    看起来很麻烦，实际上只需要先处理高个就行，每次讲最高的人插到第k个位置
    因为个子比我矮的人还没站队，队伍里都是>=我的人，我就站到第k个就行
    def reconstructQueue(self, people):
        """
        :type people: List[List[int]]
        :rtype: List[List[int]]
        """
        people.sort(key=lambda (h,k):(-h,k))
        queue = []
        for p in people:
            queue.insert(p[1], p)
        return queue
    

## 数据流中保持中位数
    数据持续涌入，还要保持中位数，显然要用到特殊的结构
    优先队列，也就是一个最小堆和一个最大堆，保持两堆大小差距在1以内
    新来一个数字，如果比最大堆(存储较小的一半)堆顶小，加入最小堆
    如果两边不平衡，最大堆堆顶弹出加入最小堆

## 高频的K个单词 *
    频率优先，频率相同则看字母顺序
    typedef pair<string, int> pp;
    struct cmp{
      bool operator()(const pp& a, const pp&b){
          if(a.second==b.second)
              return a > b;
          return a.second<b.second;
      }  
    };
    priority_queue<pp, vector<pp>, cmp> q;

## 滑动窗口最大值 
    LeetCode 239; 剑指offer59
    随着窗口滑动，持续输出当前窗口最大值
    O(n)的解法。不要记录所有值，如果新加值M比前面值大，则删掉前面比M小的(相等则不删)，他们在自己之后有生之年都不可能是当前窗口的最大值了。
    使用deque，并且保持deque内部不增。deque很适合滑动窗口问题。

# ----树----
**树的题目多少要用递归**
##前中后序遍历 
    递归

##判断二叉树是否合法 
    中序遍历，应当是递增数组

## 二叉树的最长路径
    求的是任意两点之间的最长距离
    递归，深度L+R;不断更新ans
    depth:
	    L = depth(node.left);
	    R = depth(node.right);
	    ans = max(ans, L+R+1);
	    return max(L, R) + 1;

## 最大二叉树
    LeetCode654
    由数组建树，最大为根，左右子树递归构建
    递归构建是最容易想的思路
    找寻当前最大值的位置 max_element(nums.begin(),nums.end()) - nums.begin()

##由先序遍历重构二叉搜索树 
    因为是BST，左小右大。又是先序遍历。那么比我大的都是右子树。递归恢复即可

##由先序和后序重构二叉树
    LeetCode 889
    结果不唯一。 一种解法是用双指针.当post指针走到的值与当前递归的root值相等时，说明以当前root为根的树建完了
    int preIndex = 0, posIndex = 0;
    TreeNode* constructFromPrePost(vector<int>& pre, vector<int>& post) {
        TreeNode* root = new TreeNode(pre[preIndex++]);
        if (root->val != post[posIndex])
            root->left = constructFromPrePost(pre, post);
        if (root->val != post[posIndex])
            root->right = constructFromPrePost(pre, post);
        posIndex++;
        return root;
    }

## 序列化二叉树

## 反序列化二叉树

## 判断二叉树是否平衡二叉树
    剑指offer55
    平衡即左右子树深度差不超过1
    递归判断。要求左右分支的深度。为了避免重复计算深度，先算分支深度，再后序判断是否二叉树

##字典树实现 *
     需要实现几个函数insert,search,clear;
     每个节点需要有属性isword ,以及指向26个字母的指针
     class TrieNode{
     public:
        TrieNode *next[26];
        bool isword;
    
        TrieNode(bool b=false){
            isword = b;
            memset(next, NULL, sizeof(next)); 
        }
     };
    class Trie {
		TrieNode *root;
		TrieNode * cur;
    
		public:
		    /** Initialize your data structure here. */        
		    Trie() {
		        root = new TrieNode();
		        cur = NULL;
		    }
		    
		    ~Trie(){   
		        clear(root);
		    }
		    
		    void clear(TrieNode *root){  //指针回收,后序递归删除
		        for(int i = 0; i < 26; i++){
		            if(root->next[i] != nullptr){
		                clear(root->next[i]);
		            }
		        }
		        delete root;
		    }
		    
		    /** Inserts a word into the trie. */
		    void insert(string word) {
		        cur = root;
		        int i;
		        for(i=0;i<word.size()-1;i++){
		            if(cur->next[word[i]-'a']  == NULL){
		                TrieNode *newNode = new TrieNode();
		                cur->next[word[i]-'a'] = newNode;
		                cur = newNode;
		            }else{
		                cur = cur->next[word[i]-'a'];
		            }
		        }
		        if(cur->next[word[i]-'a'] == NULL){
		            TrieNode *newNode = new TrieNode(true);
		            cur->next[word[i]-'a'] = newNode;
		        }else{
		            cur->next[word[i]-'a']->isword = true;
		        }
		    }
		    
		    /** Returns if the word is in the trie. */
		    bool search(string word) {
		        cur = root;
		        for(int i=0;i<word.size();i++){
		            if(cur->next[word[i]-'a'] == NULL){
		                return false;
		            }else{
		                cur = cur->next[word[i]-'a'];
		            }
		        }
		        return cur->isword;
		    }
        }// end of class Trie


# ----字符串----

## 替换字符串空格
    剑指offer5
    将空格换成其他字符，在C++数组中会造成前后数组长度变化
    统计空格个数和变化的总长度，设置两个指针，分别指向原字符串结尾和替换后字符串的结尾

##KMP 

## 统计回文子串的个数
    回文必有中心点，比较直接的解法O(n^2)，选择中心向两边扩张判断
    可能有一个中心点，也可能有两个 aba vs abba
    对于这种情况，优雅的考虑奇偶
    for center in xrange(2*N - 1):
        left = center / 2
        right = left + center % 2 

## 翻转单词顺序 *
    剑指offer58
    先翻转整个句子，再将每个单词翻转
    实现一个翻转字符串的函数
    void Reverse(char *begin, char *end){
          if(begin==nullptr || end==nullptr)
              return;
           while(begin < end)
           {
               char temp = *begin;
               *begin = *end;
               *end = temp;
               begin++;
               end --;
           }
      }

## 左旋转字符串
    剑指offer58， 将字符串前面几位移到后面去
    同上，分别reverse前几位和剩余位，再整体reverse

## 从字符串里能构建的最长回文串
    LeetCode 409
    其实就是数奇数个数的字符个数
    for (char c='A'; c<='z'; c++)
        odds += count(s.begin(), s.end(), c) & 1;

# ----数组与数----
##查找数组重复元素 *
    LeetCode3
    n个数字，取值于[0,n-1]，至少一个数字重复，找出其中一个
    利用取值范围，in-place交换值，努力使得i在数组第i位，如果发现第i位已经是i，则重复
    

##查找数组缺失数 *
    0-n,但是只有n个数，假定只有一个数缺失
    n(n+1)/2 - 求和
    xor -> A^A = 0
    那么最后没有消掉的就是缺失数

## 数组中有两个数字只出现1次，其他均出现2次 *
    剑指offer56
    O(n) + O(1)
    依靠xor不能区别最后两个数字，将这两个数字分组；
    先xor一遍，找到第一个为1的bit，依据该位来分组

## 数组中只有一个数字出现1次，其他均出现3次
    剑指offer56 
    对于所有出现三次的数字，其每一位之和必定能被3整除
    加起来，某一位不能被3整除，则是那个数字拥有的一位
     O(n) + O(32)

## 统计bit=1的位个数 *
     剑指offer15
     n-1 & n 可以实现将最右的1变成0
     判断有多少1 -> 是否是2的整数次
     改变2进制的多少位可以由m变成n -> 先异或再数1

## 统计数组中的逆序对
      剑指offer51
     暴力法是O(n^2)
     分组，统计一组时利用递归出的结果，再进行归并，保证统计完后也是有序的


# ----二维矩阵与查找----
排好序的数组往往适合二分查找
## 查找某个单词是否存在于一个字符矩阵中 *
    Leetcode79 类似题目很多，通常不允许走重复点
    那么在递归时，
    (1)需要记录当前已走的点，并且在递归回来时重置
    (2)引用传递，不要复制
    (3) 无脑dfs，至于是否合理（矩阵不越界）在函数初始时判断


## 矩阵查找某个数是否存在 *
    行列有序
    左下或者右上起步查询
# ----C++----
##字符串反转
    string r;
    r.assign(s);
    reverse(r.begin(), r.end());

## 高级写法
    for(auto &p:cnt)
    for(string & word: words)
    pointer == nullptr
     *p == '\0'
    

## 细节
    STL中注意通过back()等方法获取元素时要先判断容器是否为空
    右移运算符 >> 左边补位时如果是无符号用0补，如果有符号，用原先的符号位补。所有如果真想实现除2，用unsigned

# ----Python----

## Collections.Counter
    from collections import Counter
	colors = ['red', 'blue', 'red', 'green', 'blue', 'blue']
	c = Counter(colors)
	dict(c)  # 转化成字典
	
	d = {....}
	c = Counter(d)  # 字典转化为Counter
	list(c.elements())  # 重新得到colors
    Counter('abracadabra').most_common(3)  # 返回列表[(k1,v1),(k2,v2)]
    c = Counter(['eggs', 'ham'])
    c['bacon']                              # 不存在就返回0
    Counter之间可以加减交并