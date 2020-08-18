LC 221 
Maximal Square
Given a 2D binary matrix filled with 0's and 1's, find the largest square containing only 1's and return its area.

Example:

Input: 

1 0 1 0 0
1 0 1 1 1
1 1 1 1 1
1 0 0 1 0

Output: 4

-----------------------
一开始也觉得是cp，最后还是按照经典01岛屿的思路走了
很低效的方法
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        int m = matrix.size();
        if(m==0)
            return 0;
        int n = matrix[0].size();
        if(n==0)
            return 0;
        vector<pair<int,int>> c; 
        // Initialize
        for(int i=0;i<m;i++){
            for(int j=0;j<n;j++){
                if(matrix[i][j]=='1'){
                    c.push_back(make_pair(i,j));
                }
            }
        }
        if(c.empty())
            return 0;

        int step = 0;
        for(auto it=c.begin();it!=c.end();it++)
        {
            bool flag=true;
            int x = it->second;
            int y = it->first;
            while(flag){
                step ++;
                int x_max = x + step;
                int y_max = y + step;
                bool flag = true;
                if(x_max >= n || y_max >=m) // 越界
                {
                    step--;
                    break;
                }
                for(int col=x; col<=x_max; col++){
                   if(matrix[y_max][col] == '0'){
                       flag = false;
                       break; 
                   }
                }
                if(!flag){
                    step--;
                    break;
                }
                for(int row=y; row<=y_max; row++){
                    if(matrix[row][x_max] == '0'){
                        flag = false;
                        break;
                    }
                }
                if(!flag)
                    step--;
            }
            
        }  // end of for 
        step++;
        return step*step;
        
    }
};

------------------------------------------------------
正解自然是dp
dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1
dp[i][j]是到i,j位置为止的矩阵中最大正方形数
min()，就要求其左上方三个都强，它才能强
并且逐行来做的话，只需要保持当前行和上一行的dp状态，从而减少dp所使用的空间。
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if (matrix.empty()) {
            return 0;
        }
        int m = matrix.size(), n = matrix[0].size(), sz = 0;
        vector<int> pre(n, 0), cur(n, 0);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (!i || !j || matrix[i][j] == '0') {
                    cur[j] = matrix[i][j] - '0';
                } else {
                    cur[j] = min(pre[j - 1], min(pre[j], cur[j - 1])) + 1;
                }
                sz = max(cur[j], sz);
            }
            fill(pre.begin(), pre.end(), 0);
            swap(pre, cur);
        }
        return sz * sz;
    }
};













