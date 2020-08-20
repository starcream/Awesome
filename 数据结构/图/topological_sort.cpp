Leetcode 207 Course Schedule

There are a total of numCourses courses you have to take, labeled from 0 to numCourses-1.

Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]

Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?


Input: numCourses = 2, prerequisites = [[1,0],[0,1]]
Output: false
Explanation: There are a total of 2 courses to take. 
             To take course 1 you should have finished course 0, and to take course 0 you should
             also have finished course 1. So it is impossible.
             
             
    
这题一看就是拓扑排序的题
只不过不是一个连通图，可能有多个连通分支

比较好的解法(BFS)
首先只给[edges]的写法后续处理时并不方便，因此将其转化为adjacent list的形式，具体而言就是每个节点后面跟着一步能到达的点
class Solution {
public:
    bool canFinish(int numCourses, vector<pair<int, int>>& prerequisites) {
        graph g = buildGraph(numCourses, prerequisites);
        vector<int> degrees = computeIndegrees(g);    // 只统计一次入度
        for (int i = 0; i < numCourses; i++) {    // 每次尝试找一个入度为0的点
            int j = 0;
            for (; j < numCourses; j++) {
                if (!degrees[j]) {    
                    break;
                }
            }
            if (j == numCourses) {  // 每次只处理一个，如果找不到入度为0的点，说明有环
                return false;
            }
            degrees[j]--;    //   变成-1，也算删除了
            for (int v : g[j]) {   // 由j出发到达的点入度都减1
                degrees[v]--;
            }
        }
        return true;
    }
private:    // 这几个函数是私有的
    typedef vector<vector<int>> graph;    // 值得学习，不然二维vector看起来很乱
    
    graph buildGraph(int numCourses, vector<pair<int, int>>& prerequisites) {
        graph g(numCourses);
        for (auto p : prerequisites) {
            g[p.second].push_back(p.first);
        }
        return g;
    }
    
    vector<int> computeIndegrees(graph& g) {
        vector<int> degrees(g.size(), 0);
        for (auto adj : g) {
            for (int v : adj) {
                degrees[v]++;
            }
        }
        return degrees;
    }
};
    
既然是BFS，就自然可以用队列。这个问题处理好的关键在于一次只找一个入度为0的点，而不是全部
bool canFinish(int n, vector<pair<int, int>>& pre) {
    vector<vector<int>> adj(n, vector<int>());
    vector<int> degree(n, 0);
    for (auto &p: pre) {
        adj[p.second].push_back(p.first);
        degree[p.first]++;
    }
    queue<int> q;
    for (int i = 0; i < n; i++)
        if (degree[i] == 0) q.push(i);
    while (!q.empty()) {
        int curr = q.front(); q.pop(); n--;
        for (auto next: adj[curr])
            if (--degree[next] == 0) q.push(next);
    }
    return n == 0;
}
    
    
    
