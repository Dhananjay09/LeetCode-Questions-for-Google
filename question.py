'''
Given two integers dividend and divisor, divide two integers without using multiplication, division, and mod operator.

The integer division should truncate toward zero, which means losing its fractional part. For example, 8.345 would be truncated to 8, and -2.7335 would be truncated to -2.

Return the quotient after dividing dividend by divisor.

Note: Assume we are dealing with an environment that could only store integers within the 32-bit signed integer range: [−231, 231 − 1]. For this problem, if the quotient is strictly greater than 231 - 1, then return 231 - 1, and if the quotient is strictly less than -231, then return -231.

 
'''

class Solution:
    def divide(self, dividend: int, divisor: int) -> int:
        negatigve = False
        if dividend >=0 and divisor < 0:
            negatigve = True
        elif dividend <0 and divisor >= 0:
            negatigve = True
        count = 0
        divisor = abs(divisor)
        dividend = abs(dividend)
        temp = divisor
        while temp <= dividend:
            count += 1
            temp +=  divisor
        if negatigve:
            return count * -1
        return count


'''
Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.

Notice that the solution set must not contain duplicate triplets.
'''
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        ans = []
        nums.sort()
        l = len(nums)
        for i in range(0, l):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left = i + 1
            right = l - 1
            while left < right:
                total = nums[i] + nums[left] + nums[right]
                if total == 0:
                    ans.append([nums[i], nums[left], nums[right]])
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif total < 0:
                    left += 1
                else:
                    right -= 1
        return ans

'''
Given an m x n integer matrix matrix, if an element is 0, set its entire row and column to 0's.

You must do it in place.
 
Example 1:

Input: matrix = [[1,1,1],[1,0,1],[1,1,1]]
Output: [[1,0,1],[0,0,0],[1,0,1]]
Example 2:


Input: matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]
'''

class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        first_row, first_col = False, False
        
        m = len(matrix)
        n = len(matrix[0])
        
        print (matrix)
        
        for i in range(m):
            if matrix[i][0] == 0:
                first_col = True
                break
                
        for j in range(n):
            if matrix[0][j] == 0:
                first_row = True
                break
                
        for i in range(1, m):
            for j in range(1, n):
                if matrix[i][j] == 0:
                    matrix[i][0] = 0
                    matrix[0][j] = 0
                    
        for i in range(1, m):
            if matrix[i][0] == 0:
                for j in range(1, n):
                    matrix[i][j] = 0
                    
        for j in range(1, n):
            if matrix[0][j] == 0:
                for i in range(1, m):
                    matrix[i][j] = 0
        
        if first_row:
            for j in range(n):
                matrix[0][j] = 0
                
        if first_col:
            for i in range(m):
                matrix[i][0] = 0
                
'''
Given an array of strings strs, group the anagrams together. You can return the answer in any order.

 

Example 1:

Input: strs = ["eat","tea","tan","ate","nat","bat"]

Output: [["bat"],["nat","tan"],["ate","eat","tea"]]

Explanation:

There is no string in strs that can be rearranged to form "bat".
The strings "nat" and "tan" are anagrams as they can be rearranged to form each other.
The strings "ate", "eat", and "tea" are anagrams as they can be rearranged to form each other.
'''
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        dicts = {}
        answer = {}
        for word in strs:
            m_dict = {}
            for ch in word:
                if ch not in m_dict:
                    m_dict[ch] = 1
                else:
                    m_dict[ch] += 1
            found = False
            key_name = None
            for item in dicts:
                if dicts[item] == m_dict:
                    found = True
                    key_name = item
                    break
            if found:
                answer[key_name].append(word)
            else:
                answer[word] = [word]
                dicts[word] = m_dict
        return (list(answer.values()))

'''
Given a string s, find the length of the longest substring without duplicate characters.

 

Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
'''
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        st = set()
        left = 0
        ans = 0
        for right in range(0, len(s)):
            while s[right] in st:
                st.remove(s[left])
                left += 1
            st.add(s[right])
            ans = max(ans, right - left +1)
        return ans
'''
Given a string s, return the longest palindromic substring in s.

 

Example 1:

Input: s = "babad"
Output: "bab"
Explanation: "aba" is also a valid answer.
Example 2:

Input: s = "cbbd"
Output: "bb"
'''
class Solution:
    
    
    def explore(self, s, start, end):
        while start >= 0 and end<len(s):
            if s[start] == s[end]:
                start -= 1
                end += 1
            else:
                break
        return s[start+1:end]
    
    def longestPalindrome(self, s: str) -> str:
        ans = ""
        for i in range(0, len(s)):
            first = self.explore(s, i, i)
            if first and len(first)>len(ans):
                ans = first
            second = self.explore(s, i, i+1)
            if second and len(second)>len(ans):
                ans = second
        return ans

'''
Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.

'''

class Solution:
    
    def increasingTriplet(self, nums: List[int]) -> bool:
        first = second = float('inf')
        for num in nums:
            if num <= first:
                first = num
            elif num <= second:
                second = num
            else:
                return True
        return False
'''
The count-and-say sequence is a sequence of digit strings defined by the recursive formula:

countAndSay(1) = "1"
countAndSay(n) is the run-length encoding of countAndSay(n - 1).
Run-length encoding (RLE) is a string compression method that works by replacing consecutive identical characters (repeated 2 or more times) with the concatenation of the character and the number marking the count of the characters (length of the run). For example, to compress the string "3322251" we replace "33" with "23", replace "222" with "32", replace "5" with "15" and replace "1" with "11". Thus the compressed string becomes "23321511".

Given a positive integer n, return the nth element of the count-and-say sequence.
'''

ans = [1]

def construct(n):
    i =0
    n = str(n)
    l = len(n)
    ans = ""
    while(i<l):
        ch = n[i]
        count = 1
        while i<(l-1) and n[i]==n[i+1]:
            count += 1
            i+=1
        ans = ans + str(count) + ch
        i+=1
    return ans
for i in range(1, 30):
    ans.append(construct(ans[-1]))

class Solution:
    def countAndSay(self, n: int) -> str:
        return str(ans[n-1])
'''
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

'''

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        carry = 0
        ans = None
        while l1 and l2:
            total = l1.val + l2.val + carry
            carry = int(total / 10)
            total = (total - carry * 10)
            if not ans:
                ans = ListNode(total)
                response = ans
            else:
                ans.next = ListNode(total)
                ans = ans.next
            l1 = l1.next
            l2= l2.next
        
        while l1:
            total = l1.val + carry
            carry = int(total / 10)
            total = (total - carry * 10)
            if not ans:
                ans = ListNode(total)
                response = ans
            else:
                ans.next = ListNode(total)
                ans = ans.next
            l1 = l1.next
        
        while l2:
            total = l2.val + carry
            carry = int(total / 10)
            total = (total - carry * 10)
            if not ans:
                ans = ListNode(total)
                response = ans
            else:
                ans.next = ListNode(total)
                ans = ans.next
            l2 = l2.next
        
        if carry > 0:
            if not ans:
                ans = ListNode(carry)
                response = ans
            else:
                ans.next = ListNode(carry)
        return response
'''
Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even indices, and return the reordered list.

The first node is considered odd, and the second node is even, and so on.

Note that the relative order inside both the even and odd groups should remain as it was in the input.

You must solve the problem in O(1) extra space complexity and O(n) time complexity.

'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        if not head.next:
            return head
        odd = head
        even = head.next
        even_dead = even
        while even and even.next:
            odd.next = even.next
            odd = odd.next
            
            even.next = odd.next
            even =even.next
        odd.next = even_dead
            
        return head
'''
Given the heads of two singly linked-lists headA and headB, return the node at which the two lists intersect. If the two linked lists have no intersection at all, return null.

For example, the following two linked lists begin to intersect at node c1:


The test cases are generated such that there are no cycles anywhere in the entire linked structure.

Note that the linked lists must retain their original structure after the function returns.
'''
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        l1 = 0
        temp = headA
        while temp:
            l1+=1
            temp = temp.next
        l2 = 0
        temp = headB
        while temp:
            l2+=1
            temp = temp.next
        while l1>l2:
            headA = headA.next
            l1-=1
        while l2>l1:
            headB = headB.next
            l2-=1
        while(headA != headB):
            headA = headA.next
            headB = headB.next
        return headA

'''
Given the root of a binary tree, return the inorder traversal of its nodes' values.

'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    
    def __init__(self):
        self.ans = []
    def inorderTraversal(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return self.ans
        self.inorderTraversal(root.left)
        self.ans.append(root.val)
        self.inorderTraversal(root.right)
        return self.ans
'''
Given two integer arrays preorder and inorder where preorder is the preorder traversal of a binary tree and inorder is the inorder traversal of the same tree, construct and return the binary tree.
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        if not preorder or not inorder:
            return None
        root = TreeNode(preorder[0])
        
        r_index = inorder.index(preorder[0])
        
        l_i = inorder[:r_index]
        r_i = inorder[r_index+1:]
        
        len_l_i = len(l_i)
        
        l_p = preorder[1:len_l_i+1]
        r_p = preorder[len_l_i+1:]
        root.left = self.buildTree(l_p, l_i)
        root.right = self.buildTree(r_p, r_i)
        
        return root
'''
You are given a perfect binary tree where all leaves are on the same level, and every parent has two children. The binary tree has the following definition:

struct Node {
  int val;
  Node *left;
  Node *right;
  Node *next;
}
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.
'''
"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return None
        queue = [root]        
        ans = None
        while queue:
            l_queue = len(queue)
            for value in range(l_queue):
                item = queue.pop(0)
                if not ans:
                    ans = Node(item.val)
                    res = ans
                else:
                    ans.next = Node(item.val)
                    ans = ans.next
                if item.left:
                    queue.append(item.left)
                if item.right:
                    queue.append(item.right)
            if queue:
                ans.next = Node('#')
                ans = ans.next
        return res

'''
Given the root of a binary search tree, and an integer k, return the kth smallest value (1-indexed) of all the values of the nodes in the tree.
'''
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def __init__(self):
        self.ans = 0
        
    def get_list(self, root: Optional[TreeNode]) -> list:
        if not root:
            return root
        self.get_list(root.left)
        self.ans += 1
        self.get_list(root.right)
        return self.ans
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        ans = self.get_list(root)
        return ans[k-1]

'''
Given an m x n 2D binary grid grid which represents a map of '1's (land) and '0's (water), return the number of islands.

An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
'''
class Solution:
    
    def dfs(self, grid, i, j, rows, cols):
        if i < 0 or i >= rows or j < 0 or j >= cols or  grid[i][j] == '0':
            return
        grid[i][j] = '0'
        self.dfs(grid, i+1, j, rows, cols)
        self.dfs(grid, i-1, j, rows, cols)
        self.dfs(grid, i, j-1, rows, cols)
        self.dfs(grid, i, j+1, rows, cols)

    def numIslands(self, grid):
        count = 0
        rows = len(grid)
        cols = len(grid[0])
        for i in range(0, len(grid)):
            for j in range(0, len(grid[0])):
                if grid[i][j] == '1':
                    count += 1
                    self.dfs(grid, i, j, rows, cols)
        return count


'''
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent. Return the answer in any order.

A mapping of digits to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.
'''
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        result = []
        d_to_l = {
            '2': 'abc', '3': 'def', '4': 'ghi', '5': 'jkl', '6': 'mno',
            '7': 'pqrs', '8': 'tuv', '9': 'wxyz'}
        def backtrack(self, index, path):
            if index == len(digits):
                result.append("".join(path))
                return
            for letter in d_to_l[digits[index]]:
                path.append(letter)
                backtrack(self, index+1, path)
                path.pop()
            
        backtrack(self,0, [])
        return result

'''
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

'''
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        
        l = len(nums)
        
        def backtrack(path, used):
            if len(path) == l:
                result.append(path[:])
                return
            for i in range(l):
                if not used[i]:
                    used[i] = True
                    path.append(nums[i])
                    backtrack(path, used)
                    path.pop()
                    used[i] = False
        result = []
        backtrack([], [False] * len(nums))
        return result
'''
Given an integer array nums of unique elements, return all possible subsets (the power set).

The solution set must not contain duplicate subsets. Return the solution in any order.
'''
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        def backtrack(start, path):
            result.append(path[:])
            for i in range(start, len(nums)):
                path.append(nums[i])
                backtrack(i+1, path)
                path.pop()
        result = []
        backtrack(0, [])
        return result
'''
Given an array nums with n objects colored red, white, or blue, sort them in-place so that objects of the same color are adjacent, with the colors in the order red, white, and blue.

We will use the integers 0, 1, and 2 to represent the color red, white, and blue, respectively.

You must solve this problem without using the library's sort function.

'''

class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        start = 0
        end = len(nums) -1
        mid = 0
        while mid <= end:
            if nums[mid] == 1:
                mid += 1
            elif nums[mid] == 0:
                nums[mid], nums[start] = nums[start], nums[mid]
                start += 1
                mid+=1
            else:
                nums[mid], nums[end] = nums[end], nums[mid]
                end -= 1

'''
A peak element is an element that is strictly greater than its neighbors.

Given a 0-indexed integer array nums, find a peak element, and return its index. If the array contains multiple peaks, return the index to any of the peaks.

You may imagine that nums[-1] = nums[n] = -∞. In other words, an element is always considered to be strictly greater than a neighbor that is outside the array.

You must write an algorithm that runs in O(log n) time.

 
'''
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums) - 1
        while (left < right):
            mid = (left + right)//2
            if nums[mid] > nums[mid + 1]:
                right = mid
            else:
                left = mid + 1
        return left

'''
Given an array of integers nums sorted in non-decreasing order, find the starting and ending position of a given target value.

If target is not found in the array, return [-1, -1].

You must write an algorithm with O(log n) runtime complexity.
 
'''
class Solution:
    
    def binary_search(self, nums, target, is_left):
        start = 0
        end = len(nums)
        ans = -1
        while start < end:
            mid = (start + end)//2
            if nums[mid] == target:
                if is_left:
                    end = mid
                else:
                    start = mid + 1
                ans = mid
            elif nums[mid] < target:
                start = mid + 1
            else:
                end = mid
        #print (ans)
        return ans
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        
        return [self.binary_search(nums, target, True), self.binary_search(nums, target, False)]

'''
Given an array of intervals where intervals[i] = [starti, endi], merge all overlapping intervals, and return an array of the non-overlapping intervals that cover all the intervals in the input.
'''
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        intervals.sort(key=lambda x: x[0])
        ans = [intervals[0]]
        for start, end in intervals[1:]:
            last_end = ans[-1][1]
            if start<=last_end:
                ans[-1][1] = max(end, last_end)
            else:
                ans.append([start, end])
        return ans
'''
  Search a 2D Matrix II

Solution
Write an efficient algorithm that searches for a value target in an m x n integer matrix matrix. This matrix has the following properties:

Integers in each row are sorted in ascending from left to right.
Integers in each column are sorted in ascending from top to bottom.
'''
class Solution:
    
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        rows = len(matrix)
        cols = len(matrix[0])
        row = 0
        col = cols - 1
        while row < rows and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        return False

'''
You are given an integer array nums. You are initially positioned at the array's first index, and each element in the array represents your maximum jump length at that position.

Return true if you can reach the last index, or false otherwise.

'''
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        max_reach = 0
        l = len(nums)
        for i in range(l):
            if i > max_reach:
                return False
            max_reach = max(max_reach, i+nums[i])
            if max_reach >= l:
                return True
        return True
'''
There is a robot on an m x n grid. The robot is initially located at the top-left corner (i.e., grid[0][0]). The robot tries to move to the bottom-right corner (i.e., grid[m - 1][n - 1]). The robot can only move either down or right at any point in time.

Given the two integers m and n, return the number of possible unique paths that the robot can take to reach the bottom-right corner.

The test cases are generated so that the answer will be less than or equal to 2 * 109.

'''
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        arr = []
        for i in  range(0, m):
            lst = []
            for j in range(0, n):
                lst.append(1)
            arr.append(lst)
        for i in range(1, m):
            for j in range(1, n):
                arr[i][j] = arr[i][j-1] + arr[i-1][j] 
        return (arr[m-1][n-1])

'''
You are given an integer array coins representing coins of different denominations and an integer amount representing a total amount of money.

Return the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

You may assume that you have an infinite number of each kind of coin.
'''
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount+1):
                dp[i] = min(dp[i], dp[i-coin] + 1)
        return dp[amount] if dp[amount] != amount+1 else -1
'''
Given an integer array nums, return the length of the longest strictly increasing subsequence.
'''
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        dp = [1] * n
        for i in range(n):
            for j in range(i):
                if nums[i]>nums[j]:
                    dp[i] = max(dp[i], dp[j]+1)
        return max(dp)

'''
Write an algorithm to determine if a number n is happy.

A happy number is a number defined by the following process:

Starting with any positive integer, replace the number by the sum of the squares of its digits.
Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
Those numbers for which this process ends in 1 are happy.
Return true if n is a happy number, and false if not.

'''
class Solution:
    
    def cal(self, n):
        ans = 0
        for item in str(n):
            ans += int(item) * int(item)
        return ans
    def isHappy(self, n: int) -> bool:
        slow, fast = n, self.cal(n)
        while (fast != 1 and slow != fast):
            slow = self.cal(slow)
            fast = self.cal(self.cal(fast))
        return fast == 1

'''
Given an integer n, return the number of trailing zeroes in n!.

Note that n! = n * (n - 1) * (n - 2) * ... * 3 * 2 * 1.
'''
class Solution:
    def trailingZeroes(self, n: int) -> int:
        count = 0
        while n >= 5:
            count += n//5
            n //= 5
        return count

'''
Given a string columnTitle that represents the column title as appears in an Excel sheet, return its corresponding column number.

For example:
'''
class Solution:
    def titleToNumber(self, columnTitle: str) -> int:
        ans = 0
        for ch in columnTitle:
            ans = ans *26 + (ord(ch) - ord('A') + 1)
        return ans
