Leetcode 41  First Missing positive
利用数组索引当hash 
 
 Given an unsorted integer array, find the smallest missing positive integer.

Example 1:

Input: [1,2,0]
Output: 3
Example 2:

Input: [3,4,-1,1]
Output: 2
Example 3:

Input: [7,8,9,11,12]
Output: 1
Follow up:

Your algorithm should run in O(n) time and uses constant extra space.
 我的解法，类似于找0-N-1中重复数，利用数组本身当索引，不用额外空间，O(n)时间
 唯一陷阱就是，交换数的时候，如果两数相等，可能会无限循环
 class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        /*  最小的缺失正数，最小只可能是1
        最大就是 n+1 。可以参考剑指offer找重复数的方法*/
        int n = nums.size();
        for(int i=0;i<n;i++){
            while(nums[i] != i+1)
            {
                if(nums[i]<=0 || nums[i]>n){
                    nums[i] = -1;
                    break;
                }

                int tmp = nums[i];
                if(tmp==nums[tmp-1])  // same value, don't swap
                    break;
                nums[i] = nums[tmp-1]; 
                nums[tmp-1] = tmp;
            }
        }
        for(int i=0;i<n;i++){
            if(nums[i]!=i+1)
                return i+1;
        }
        return n+1;
    }
};
 
python的解法，统计频率，出现过的位置就加n
 
 
 def firstMissingPositive(self, nums):
    """
    :type nums: List[int]
    :rtype: int
     Basic idea:
    1. for any array whose length is l, the first missing positive must be in range [1,...,l+1], 
        so we only have to care about those elements in this range and remove the rest.
    2. we can use the array index as the hash to restore the frequency of each number within 
         the range [1,...,l+1] 
    """
    nums.append(0)
    n = len(nums)
    for i in range(len(nums)): #delete those useless elements
        if nums[i]<0 or nums[i]>=n:
            nums[i]=0
    for i in range(len(nums)): #use the index as the hash to record the frequency of each number
        nums[nums[i]%n]+=n
    for i in range(1,len(nums)):
        if nums[i]/n==0:
            return i
    return n
