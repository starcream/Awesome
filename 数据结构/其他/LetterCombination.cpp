Leetcode 17
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.
就是组合所有字符串的组合

我的解法:   注意，用迭代。如果递归，很可能会有很多重复计算。这段代码时间okay，空间效率还是差点，也有诸多不足
其中也有一些细节：
int('1') = 49
string s('a') 会报错
string s(1, 'a')

class Solution {
public:
    map<int, string> mapping={
        {2, "abc"},
        {3, "def"},
        {4, "ghi"},
        {5, "jkl"},
        {6, "mno"},
        {7, "pqrs"},
        {8, "tuv"},
        {9, "wxyz"}
    };
    vector<string> helper(vector<string> &ans, char &digit){
        int number = int(digit)-48;
        vector<string> result;
        if(ans.size()==0){   // empty
            for(auto &ch : mapping[number]){
                string s(1, ch);
                result.push_back(s);
            }
            return result;
        }
     
        for(auto &str : ans){
            for(auto &ch : mapping[number]){
                result.push_back(str + ch);
            }
        }
        return result;
    }
    
    vector<string> letterCombinations(string digits) {
        vector<string> ans;
        for(auto &ch : digits){
            if(ch=='0' || ch=='1'){ vector<string> p; return p;}
            ans = helper(ans, ch);
        }
        return ans;
    }
};


更优的解法。 首先是不使用class的成员变量，而是在函数内部设置常量 static const
其次是使用一个初始化的seed，避免开始无法组合字符串的情况
最后是使用一个swap方法，因为前一次的结果用完之后是不需要保存的，所有每次swap给result
vector<string> letterCombinations(string digits) {
    vector<string> result;
    if(digits.empty()) return vector<string>();
    static const vector<string> v = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    result.push_back("");   // add a seed for the initial case
    for(int i = 0 ; i < digits.size(); ++i) {
        int num = digits[i]-'0';
        if(num < 0 || num > 9) break;
        const string& candidate = v[num];
        if(candidate.empty()) continue;
        vector<string> tmp;
        for(int j = 0 ; j < candidate.size() ; ++j) {
            for(int k = 0 ; k < result.size() ; ++k) {
                tmp.push_back(result[k] + candidate[j]);
            }
        }
        result.swap(tmp);
    }
    return result;
}
