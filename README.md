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
    LeetCode 215
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
    LeetCode92 反转中间m-n的节点
    走到m，记录断点。然后用普通的链表反转，方向存在问题，需要重连一下两边
    ListNode* prev = nullptr, *nextTmp=nullptr, *breakpoint=dummy;   
    while(head){
        count += 1;
        if(m != 1 and count == m - 1){
            breakpoint = head;
        }
        if(count >= m && count <=n){
             // 这里同链表反转
            nextTmp = head->next;
            head->next = prev;
            prev = head;
            head = nextTmp;
        }
        else if (count < m){
            head = head->next;
        }else{
            break;
        }
    }
    breakpoint->next->next = head; //此时的head在n后一位
    breakpoint->next = prev;   // 此时的prev应该移到breakpoint后
    return dummy->next;

## 链表排序
    leetcode148
    quicksort || merge sort
    (在链表排序中，很难用random的pivot，所以效果会差一点）
    (而在链表中，归并排序不存在数组重新复制的问题，链表的合并是如此方便和高效)
	ListNode merge(ListNode list1, ListNode list2) {
        ListNode dummyHead = new ListNode();
        ListNode tail = dummyHead;
        while (list1 != null && list2 != null) {
            if (list1.val < list2.val) {
                tail.next = list1;
                list1 = list1.next;
                tail = tail.next;
            } else {
                tail.next = list2;
                list2 = list2.next;
                tail = tail.next;
            }
        }
        tail.next = (list1 != null) ? list1 : list2;
        return dummyHead.next;


## 旋转链表
    leetcode61  就是讲链表最右边k个移到前面来。比较标准的写法
	ListNode* rotateRight(ListNode* head, int k) {
	    if (!head || !head->next || k == 0) return head;
	    ListNode *cur = head;
	    int len = 1;
	    while (cur->next && ++len) cur = cur->next;  
	    cur->next = head;   // 尾部连接头，构成环
	    k = len - k % len;
	    while (k--) cur = cur->next;
	    head = cur->next;   // cur是断点
	    cur->next = nullptr;
	   return head; 
	}

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

## 股票买卖(限制交易次数，无冷却)
     LeetCode188
      1. 算是贪婪法
     找出所有的不相交上升区间，如果无交易次数限制，低点买入高点售出就行
     可惜有次数限制，因此可能需要删除上升区间或者合并相邻的上升区间。注意边界情况，第一个低谷点和最后一个峰点。注意股票价格不变的情况，低谷点一定要比后一天小；峰点一定要比前一天大，不在乎后一天是否等。注意删除vector时，利用
     v.erase(v.begin()+idx)
     时间复杂度  ：  最差O(n^2)
      2. DP
      dp[i][j] 截止第j天进行i次交易能够得到的最大利润
      如果第j天不卖， dp[i][j] = dp[i][j-1]
      如果第j天卖了，前面必然有一天t买了，dp[i][j] = max(dp[i-1][t-1] + price[j]-price[t])
      这样的复杂度 O(k) * O(n) * O(n)
      不过在j天循环的维度，可以利用dp[i-1][j-1]-price[j]作为后来者可以利用的dp[i-1][t-1] - price[t]
      for i : 1 -> k
      maxTemp = -prices[0];
	      for j : 1 -> n-1
	         dp[i][j] = max(dp[i][j-1], prices[j]+maxTemp);
	         maxTemp = max(maxTemp, dp[i-1][j-1]-prices[j]); // 第j天买入 
      return dp[k][n-1];


## 目标和
      LeetCode494
      给定数组，允许用+和-,求有多少种达到目标和的方式  
      很容易联想到递归--有记忆递归--自底向上DP  dp[index][sum]

## House Robber
    LeetCode 198 & 213 & 337
    前两题算是dp。规定小偷不能偷相邻两家的钱，最多偷多少钱
    就按照偷不偷此前一家来  pp,p,cur表示截止上上一家，上一家，当前一家最多偷多少钱
    cur = max(pp+nums[i], p);
    pp = p;
    p = cur;
    第三题有点难度，换成二叉树的格式
    要避免重复递归，但是右很难自底向上dp
    解法是，按照偷不偷当前节点，返回两个最大值
    pair<int, int> visit(TreeNode* root){
        // return two vals, the maximum of this tree (visiting the root or not)
        if(!root){
            return {0,0};
        }
        pair<int, int> l,r;
        l = visit(root->left);
        r = visit(root->right);
        return {root->val+l.second+r.second, max(l.first,l.second)+max(r.second,r.first)};
    }


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
    LeetCode 692
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

    // 高级的写法
    auto comp = [&](const pair<string,int>& a, const pair<string,int>& b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        };
    typedef priority_queue< pair<string,int>, vector<pair<string,int>>, decltype(comp) > my_priority_queue_t;
    my_priority_queue_t  pq(comp);
    
    for(auto w : freq ){
        pq.push({w.first, w.second});
        if(pq.size()>k) pq.pop();
    }
    


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

## 逐层打印二叉树
    LeetCode 32
    逐层 -> 队列
    之字形 -> 两个栈，每隔一轮换一次添加子节点的方向
     stack <BinaryTreeNode*> levels[2];
     if(levels[current].empty())
         current = 1 - current;

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

## 合法括号
    常规的合法括号匹配可以用栈
    这里看到一道 * 可以匹配 (,),或空的
    LeetCode 678
    如果基于*进行递归，会超时
    下面的方法，cmax表示最多能对上多少右括号，cmin表示必须要对上多少右括号。最后一定要求，cmin在0，cmax>=0
    bool checkValidString(string s) {
        int cmin = 0, cmax = 0;
        for (char c : s) {
            if (c == '(')
                cmax++, cmin++;
            if (c == ')')
                cmax--, cmin = max(cmin - 1, 0); // 右括号不能领先，因此cmin不能透支，cmax如果小于0立即报错
            if (c == '*')  // *调节作用，让max冲得更大，让min更接近0
                cmax++, cmin = max(cmin - 1, 0);  
            if (cmax < 0) return false;
        }
        return cmin == 0;
    }

## 重复DNA(字符串)序列
    子串长度固定，找重复的子串
     def findRepeatedDnaSequences(self, s: str) -> List[str]:
        # turn all possible string into int and sort，用2个bit去表示一个数
        length = len(s)
        d = []
        r = {}
        for i in range(0, len(s)-9):
            cur = 1
            for j in range(0, 10):
                if s[i+j]=='A':
                    cur<<=2
                if s[i+j]=='C':
                    cur<<=2
                    cur+=1
                if s[i+j]=='G':
                    cur<<=2
                    cur+=2
                if s[i+j]=='T':
                    cur<<=2
                    cur+=3
            d.append(cur)
            r[cur]=i
        d.sort()
        c = Counter(d)
        return [s[r[x]:r[x]+10] for x,v in c.items() if v>1]

      C++版本，用三个bit去表示一个数。 因为在ASCII表中，A is 0101, C is 0103, G is 0107, T is 0124. The last digit in octal are different for all four letters. That's all we need!  ch & 7 ==> 就可以直接区分四个字母，不需要if，else
      	vector<string> findRepeatedDnaSequences(string s) {
	    unordered_map<int, int> m;
	    vector<string> r;
	    int t = 0, i = 0, ss = s.size();
	    while (i < 9)   // 初始化t
	        t = t << 3 | s[i++] & 7;
	    while (i < ss)
	        if (m[t = t << 3 & 0x3FFFFFFF | s[i++] & 7]++ == 1)  // 掩码设置为30位是因为，最左两位是不用的，每次存10个数，30位；左移之后先损失一位，再消掉2位，相当于去掉一个数；通过并在加上一个数
	            r.push_back(s.substr(i - 10, 10));
	    return r;
	}



## 字符串去重并使得排列最小
    LeetCode 316
    "cdadabcc" => "adbc"
    // 总感觉偏贪婪算法，说实话没做出来
    string removeDuplicateLetters(string s) {
        vector<int> dict(256, 0);
        vector<bool> visited(256, false);
        for(auto ch : s)  dict[ch]++;
        string result = "0";   // 避免result.back()问题
        /** the key idea is to keep a monotically increasing sequence **/
        for(auto c : s) {
            dict[c]--;
            /** to filter the previously visited elements **/
            if(visited[c])  continue;
            while(c < result.back() && dict[result.back()]) {   
                visited[result.back()] = false;   //只要后面还有你们，并且你们优先级低于我，则我要顶替
                result.pop_back();
            }
            result += c;    // 字符串直接与char拼接
            visited[c] = true;
        }
        return result.substr(1);  // 字符串截取
    }

    另一种方法,每次加一个字符，符合要求的最小字符；
    def removeDuplicateLetters(self, s):
    for c in sorted(set(s)):
        suffix = s[s.index(c):]
        if set(suffix) == set(s):  // 说明前面的没用了，前面都比我大并且在后缀中还有
            return c + self.removeDuplicateLetters(suffix.replace(c, ''))  // 我也不要干扰后面的字符 
    return ''

# ----Mark一下贪婪----
	Any problem can be solved using dp. Solving using a greedy strategy is harder though, since you need to prove that greedy will work for that problem. There are some tell-tale signs of a problem where greedy may be applicable, but isn't immediately apparent. Example:
	
	Choice of an element depends only on its immediate neighbours (wiggle sort).
	Answer is monotonically non-decreasing or non-increasing (sorting). This is also applicable for LIS for example.
	Anything that requires lexicographically largest or smallest of something.
	Anything where processing the input in sorted order will help.
	Anything where processing the input in forward or reverse (as given) will help.
	Anything which requires you to track the minimum or maximum of something (think of sliding window problems).
	There's matroid theory which deal with greedy algorithms, but I don't really understand it. If someone does, I'll be super grateful to them to explain it to me in simple language!
	
	In general, try to see if for a problem, the solution doesn't depend on a lot of history about the solution itself, but the next part of the solution is somewhat independent from the rest of the solution. These are all indicative of the fact that a greedy strategy could be applicable




# ----数组与数----
##查找数组重复元素 *
    LeetCode3 & 287
    n个数字，取值于[0,n-1]，至少一个数字重复，找出其中一个
    (1) 利用取值范围，in-place交换值，努力使得i在数组第i位，如果发现第i位已经是i，则重复  O(n)
    // (2) 求和--溢出风险  // (3) 排序 O(nlgn)
    // (4) O(n) & O(1) & 不修改数组
    将其看成一个找链表环入口的问题
    287题中， 规定 数来自[1-N], 共有N+1个数
    我们把i->N[i]看成链表里的next
    没有人指向0，因此0是天然的起点
    int slow = nums[0];
	int fast = nums[nums[0]];  // 起点是0，这里应该是各自走了一回
	while (slow != fast)
	{
		slow = nums[slow];
		fast = nums[nums[fast]];
	}

	fast = 0;
	while (fast != slow)
	{
		fast = nums[fast];
		slow = nums[slow];
	}
	return slow;
    // (5)  鸽巢原理，二分查找
    low = 1
    high = len(nums)-1
    
    while low < high:
        mid = low+(high-low)/2
        count = 0
        for i in nums:
            if i <= mid:
                count+=1
        if count <= mid:
            low = mid+1
        else:
            high = mid
    return low


## 删除有序数组中的重复元素(每个元素至多保留k个)
      in_place修改
      [1,1,1,2,2,3,3,3,4]
      int removeDuplicates(vector<int>& nums) {
      int i = 0;
      for (int n : nums)
          if (i < k || n > nums[i-k])
              nums[i++] = n;
       return i;
      }
      

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

## 求无符号数的补数
     LeetCode 476
     补数可以用 ~  .但是会造成前面都是1
      00000101 -> 11111010 
      我们要的是00000010
      所以要使前5位区分开
      int findComplement(int num) {
        unsigned mask = ~0;
        while (num & mask) mask <<= 1;   // num&mask找到原数有多少位
        return ~mask & ~num;

        num          = 00000101
		mask         = 11111000
		~mask & ~num = 00000010
      

## 统计数组中的逆序对
      剑指offer51
     暴力法是O(n^2)
     分组，统计一组时利用递归出的结果，再进行归并，保证统计完后也是有序的

## 组合和 *
    Leetcode 39   
    Target Sum ,有点接近树的路径和
    设计一个path，递归时push，递归后pop，所谓回溯
    LeetCode40  
    数组有重复数字，但每个最多只能用一次，并且最后的结果得unique
    int prev = -1;
    for(int i=index-1;i>=0;i--){
        if(candidates[i]<=target && candidates[i]!=prev){
            path.push_back(candidates[i]);
            prev = candidates[i];
            addPath(candidates, ans, path, target-candidates[i], i);
            path.pop_back();
        }
    }

    LeetCode377  [1,2]和[2,1]算不同的组合，其实是permutation。但是本题只需要个数，不要求列出所有
    一维dp  O(nM)
    def combinationSum4(self, nums, target):
        nums, combs = sorted(nums), [1] + [0] * (target)
        for i in range(target + 1):
            for num in nums:
                if num  > i: break
                combs[i] += combs[i - num]
        return combs[target]
    这个方法看起来优雅，避免了递归，但是计算了很多无用的comb值，而且其中一些target会造成整数上溢
    自上而下的方法则是有的放矢

## 去除被覆盖的interval
     LeetCode 1288   [1,5] [2,4]  => [1,5]
     第一位升序 ，第二位降序  [1,5] [1,3] [2,3]
     那么作为后来者的你相要被保留下来，必须比前面的右边界大，否则必定被覆盖
     def removeCoveredIntervals(self, A):
        res = right = 0
        A.sort(key=lambda a: (a[0], -a[1]))
        for i, j in A:
            res += j > right
            right = max(right, j)
        return res

## 旋转数组
    LeetCode 189
    方法1：O(1)空间，O(n)时间，in_place
    逐个向右移k，必然会移到开始点。此时不一定移完了所有，需要右移开始点
    int count = 0 ,start_pos=0, next_pos=0, prev=0, tmp=0;
    for(int i=0; count < n; i++){
        start_pos = i, next_pos = start_pos+k, prev = nums[start_pos];
        while(start_pos!=next_pos){
            tmp = nums[next_pos];
            nums[next_pos] = prev; 
            prev = tmp;
            next_pos += k;
            next_pos %= n;
            count ++;
        }
        nums[next_pos] = prev;  // 回到开始点后，不要忘了给开始点赋新值
        count ++;
    }
    方法2：
    同字符串的左旋转，先reverse前后相应部分，再整体reverse；也可以先整体reverse，再部分reverse


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

## 行列各种排序的矩阵查找第k大的数 *
    一个重要的函数 查找小于等于自己的数
    def getLessEqual(mid):
        i = n-1
        j = 0
        ans = 0
        while i >=0 and j <= n-1:
            if matrix[i][j] > mid:
                i -= 1;
            else:
                ans += (i+1)  (第j列往上都比我小)
                j += 1
        return ans

     基于这个函数二分查找
     O(nlogn)
    lo = matrix[0][0]
    hi = matrix[n-1][n-1]
    
    while lo <= hi:
        mid = int(lo + (hi-lo)/2)
        if getLessEqual(mid) < k:
            lo = mid+1
        else:
            hi = mid-1
    return lo
    为何lo一定是矩阵中的元素呢
    因为当循环，也就是二分查找结束时，hi=lo-1
    小于等于hi的不足k，而小于等于lo的>=k
    那显然矩阵中存在k


## 孤岛问题 *
    LeetCode 130 Surrounded Regions 稍微麻烦一点，要求将所有closed islands反转，不closed的不变。正确解法应该从边缘岛屿入手，先将其变为'*'。然后将剩余岛屿反转。最后将'*'变回岛屿。
    LeetCode 200 Number of islands 
       BFS/DFS 注意及时将遍历过的点更改颜色; 注意如果使用DFS递归，写起来更方便，因为可以将边界判断放到函数头部
    LeetCode 1254 Number of closed islands  同上，easy
    LeetCode 695 Max Area of island  同上，easy
    LeetCode 934 Shortest Bridge  确定有两个岛屿，最少需要建多长的桥才能将其连接
    将其中一个岛屿变色并不难，如果对两个岛屿之间各点两两求距离，慢，期望是O(n^2)，即便两岛离得很近，只要岛屿大，都要计算很多次
    通过膨胀（expand）其中一个岛屿来找寻最短距离


    bool expand(vector<vector<int>>& A, int i, int j, int cl) {
	    if (i < 0 || j < 0 || i == A.size() || j == A.size()) return false;
	    if (A[i][j] == 0) A[i][j] = cl + 1;
	    return A[i][j] == 1;
    }  // 一个岛为1，一个岛为2，
    int shortestBridge(vector<vector<int>>& A) {
	    for (int i = 0, found = 0; !found && i < A.size(); ++i)
	        for (int j = 0; !found && j < A[0].size(); ++j) found = paint(A, i, j);
	    
	    for (int cl = 2; ; ++cl)  膨胀2岛
	        for (int i = 0; i < A.size(); ++i)
	            for (int j = 0; j < A.size(); ++j) 
	                if (A[i][j] == cl && ((expand(A, i - 1, j, cl) || expand(A, i, j - 1, cl) || 
	                    expand(A, i + 1, j, cl) || expand(A, i, j + 1, cl))))
	                        return cl - 2;
	}

    LeetCode 463 island Perimeter

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
    
## 定制*Compare*
    凡是有排序的地方往往都可能需要定制Compare
    set/map/priority_queue/sort/upper_bound/nth_element/min_element

    struct Point { double x, y; };
    struct PointCmp {
	    bool operator()(const Point& lhs, const Point& rhs) const { 
	        return std::hypot(lhs.x, lhs.y) < std::hypot(rhs.x, rhs.y); 
	    }
    };

    set<Point, PointCmp> z = {{2, 5}, {3, 4}, {1, 1}};

## 细节
    (1)STL中注意通过back()等方法获取元素时要先判断容器是否为空
    (2)右移运算符 >> 左边补位时如果是无符号用0补，如果有符号，用原先的符号位补。所有如果真想实现除2，用unsigned
    (3) 链表头部加入一个dummy-head
        ListNode N(0);
        ListNode * dummy = & N;
        dummy->next = head;


## STL
   [https://blog.csdn.net/weixin_38513406/article/details/108079144](https://blog.csdn.net/weixin_38513406/article/details/108079144 "STL")
    
## 稳定排序
    vector<string> words{"for","first",
							      "reinterpret","world","element","as"};
    //对words中的元素按照长度进行排序，具有相同长度的元素相对位置不会改变。
    stable_sort(words.begin(), words.end(), 
                 [](const string& s1, const string& s2)
                 { return s1.size() < s2.size(); });
## 条件查询
    int sz =5;   // 找到第一个长度大于5的单词
    auto divide = std::find_if(words.begin(), words.end(),
                           [sz](const string& s1){ 
                               return s1.size() > sz;
                           });



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