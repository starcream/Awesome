LC239 
Given an array nums, there is a sliding window of size k which is moving from the very left of the array to the very right. You can only see the k numbers in the window. Each time the sliding window moves right by one position. Return the max sliding window.

Follow up:
Could you solve it in linear time?

Example:

Input: nums = [1,3,-1,-3,5,3,6,7], and k = 3
Output: [3,3,5,5,6,7] 
Explanation: 

Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7

------------
用multiset可解，空间复杂度为O(k),时间复杂度为O(nlogk)
但是在其中可以发现，有一些数是被其他数压制（淹没了的），其实不需要存储以及参与排序
更优的O(n)解法，设置deque，很适合滑动窗口问题。并且保持deque内部不增。
如果新加元素M比之前的部分元素大，则删掉之前的，因为有M在，他们永无出头之日
而如果M小或者等，都要加入队列，因为他还是有一定机会的
这是一个后浪拍前浪的故事啊
细节：dq.back()等需要dq不空
因为定期要删除之前的最大值，所以加入新元素时，如果和之前某个值T相等，是不删T的。尊重一下同level的前浪，否则会删错
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        // deque , keep it of descending order
        vector<int> ans;
        deque<int> dq;
        for(int i=0;i<nums.size();i++){
            if(i>=k && nums[i-k]==dq.front())  {
                dq.pop_front();  
            }
            while(!dq.empty() && dq.back()<nums[i])
                dq.pop_back();
            dq.push_back(nums[i]);
            if(i>=k-1)
                ans.push_back(dq.front());
        }
        return ans;
    }
};
