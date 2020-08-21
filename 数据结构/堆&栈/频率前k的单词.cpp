LC 692. Top K Frequent Words
Given a non-empty list of words, return the k most frequent elements.

Your answer should be sorted by frequency from highest to lowest. If two words have the same frequency, then the word with the lower alphabetical order comes first.

Example 1:
Input: ["i", "love", "leetcode", "i", "love", "coding"], k = 2
Output: ["i", "love"]
Explanation: "i" and "love" are the two most frequent words.
    Note that "i" comes before "love" due to a lower alphabetical order.
Example 2:
Input: ["the", "day", "is", "sunny", "the", "the", "the", "sunny", "is", "is"], k = 4
Output: ["the", "is", "sunny", "day"]
Explanation: "the", "is", "sunny" and "day" are the four most frequent words,
    with the number of occurrence being 4, 3, 2 and 1 respectively.
------------------------------------------------
基本都是用优先队列。主要在于cmp的写法。我的解法，
class Solution {
public:
    typedef pair<string, int> pp;
    struct cmp{
      bool operator()(const pp& a, const pp&b){
          if(a.second==b.second)
              return a > b;
          return a.second<b.second;
      }  
    };
    vector<string> topKFrequent(vector<string>& words, int k) {
        unordered_map<string, int> cnt;
        for(string &word:words){    
            if(cnt.find(word)!=cnt.end())
                cnt[word] ++;
            else
                cnt[word] = 1;
        }
        priority_queue<pp, vector<pp>, cmp> q;
        vector<string> ans;
        for(auto &p:cnt)
            q.push(p);
        for(int i=0;i<k;i++)
        {
            ans.push_back(q.top().first);
            q.pop();
        }
        return ans;
    }
};
-----------------------------------------------------
我的解法将全部字符串加入队列，实际不需要，保持队列大小始终在k就好。所以我的是nlog(n),人家的是nlog(k)
class Solution {
public:
    vector<string> topKFrequent(vector<string>& words, int k) {
        unordered_map<string, int> freq;
        for(auto w : words){
            freq[w]++;
        }
        auto comp = [&](const pair<string,int>& a, const pair<string,int>& b) {
            return a.second > b.second || (a.second == b.second && a.first < b.first);
        };     //  小的在前面，在top;我只维持k大小的队列，所以小的就pop掉，留下的都是大的
        typedef priority_queue< pair<string,int>, vector<pair<string,int>>, decltype(comp) > my_priority_queue_t;
        // cmp部分必须是一个type，因此不能直接填comp，而是使用decltype先填在那
        my_priority_queue_t  pq(comp);
        
        for(auto w : freq ){
            pq.emplace(w.first, w.second);
            if(pq.size()>k) pq.pop();
        }
        
        vector<string> output;
        while(!pq.empty()){
            output.insert(output.begin(), pq.top().first);
            pq.pop();
        }
        return output;
    }
};
