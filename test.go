package main

import (
	"math"
	"sort"
	"strconv"
	"strings"
)

// 堆盘子
/*type StackOfPlates struct {
	Capacity int
	Data     [][]int
}

func Constructor(capacity int) StackOfPlates {
	return StackOfPlates{
		Capacity: capacity,
		Data:     make([][]int, 0),
	}
}

func (s *StackOfPlates) Push(val int) {
	if s.Capacity == 0 {
		return
	}

	if len(s.Data) == 0 {
		data := make([]int, 0)
		data = append(data, val)
		s.Data = append(s.Data, data)
		return
	}

	lastPlate := s.Data[len(s.Data)-1]
	if len(lastPlate) == 0 {
		data := make([]int, 0)
		data = append(data, val)
		s.Data = append(s.Data, data)
		return
	}

	if len(lastPlate) == s.Capacity {
		data := make([]int, 0)
		data = append(data, val)
		s.Data = append(s.Data, data)
		return
	}

	lastPlate = append(lastPlate, val)
	s.Data[len(s.Data)-1] = lastPlate
}

func (s *StackOfPlates) Pop() int {
	if len(s.Data) == 0 {
		return -1
	}

	lastPlate := s.Data[len(s.Data)-1]
	val := lastPlate[len(lastPlate)-1]
	newPlate := lastPlate[0 : len(lastPlate)-1]
	s.Data[len(s.Data)-1] = newPlate
	if len(newPlate) == 0 {
		s.Data = s.Data[0 : len(s.Data)-1]
	}

	return val
}

func (s *StackOfPlates) PopAt(index int) int {
	if index >= len(s.Data) {
		return -1
	}

	plate := s.Data[index]
	val := plate[len(plate)-1]
	plate = plate[0 : len(plate)-1]
	s.Data[index] = plate
	if len(plate) == 0 {
		slice := s.Data[index+1:]
		s.Data = s.Data[0:index]
		s.Data = append(s.Data, slice...)
	}

	return val
}*/

// 日程表安排，已有区间[a, b)，所给区间的左值需要小于b, 右值大于a
/*type MyCalendar struct {
	Calendar [][]int
}

func Constructor() MyCalendar {
	return MyCalendar{
		Calendar: make([][]int, 0),
	}
}

func (myCalendar *MyCalendar) Book(start int, end int) bool {
	if len(myCalendar.Calendar) == 0 {
		myCalendar.Calendar = append(myCalendar.Calendar, []int{start, end})
		return true
	}

	for _, calendar := range myCalendar.Calendar {
		if start < calendar[1] && end > calendar[0] {
			return false
		}
	}

	myCalendar.Calendar = append(myCalendar.Calendar, []int{start, end})
	return true
}*/

// 单词频率
/*type WordsFrequency struct {
	Frequency map[string]int
}

func Constructor(book []string) WordsFrequency {
	wordsFrequency := make(map[string]int)
	for _, i := range book {
		if _, ok := wordsFrequency[i]; ok {
			wordsFrequency[i]++
		} else {
			wordsFrequency[i] = 1
		}
	}

	return WordsFrequency{
		Frequency: wordsFrequency,
	}
}

func (wordsFrequency *WordsFrequency) Get(word string) int {
	if counted, ok := wordsFrequency.Frequency[word]; ok {
		return counted
	}

	return 0
}

// 最大连续子数组之和
func maxSubArray(nums []int) int {
	n := len(nums)
	// 这里的dp[i] 表示，最大的连续子数组和，包含num[i] 元素
	dp := make([]int, n)
	// 初始化，由于dp 状态转移方程依赖dp[0]
	dp[0] = nums[0]
	// 初始化最大的和
	mx := nums[0]
	for i := 1; i < n; i++ {
		// 这里的状态转移方程就是：求最大和
		// 会面临2种情况，一个是带前面的和，一个是不带前面的和
		dp[i] = max(dp[i-1]+nums[i], nums[i])
		mx = max(mx, dp[i])
	}
	return mx
}

// 找峰值数组
func findPeaks(mountain []int) []int {
	var result []int
	if len(mountain) <= 2 {
		return result
	}

	for i := 1; i < len(mountain)-1; i++ {
		if mountain[i] > mountain[i-1] && mountain[i] > mountain[i+1] {
			result = append(result, i)
			// i是峰值，i+1必不是
			i++
		}
	}

	return result
}

// 字符串的好分割
func numSplits(s string) int {
	left := make([]int, 26)
	right := make([]int, 26)
	var leftCount, rightCount, result int
	for i := 0; i < len(s); i++ {
		if right[s[i]-'a'] == 0 {
			rightCount++
		}

		right[s[i]-'a']++
	}

	for i := 0; i < len(s); i++ {
		if left[s[i]-'a'] == 0 {
			leftCount++
		}

		left[s[i]-'a']++
		right[s[i]-'a']--
		if right[s[i]-'a'] == 0 {
			rightCount--
		}

		if leftCount == rightCount {
			result++
		}
	}

	return result
}

// 回文子串数目
func countSubstrings(s string) int {
	var result int
	if len(s) == 0 {
		return result
	}

	for i := 0; i < 2*len(s)-1; i++ {
		l, r := i/2, i/2+i%2
		for l >= 0 && r < len(s) && s[l] == s[r] {
			l--
			r++
			result++
		}
	}

	return result
}

// 矩阵置0
func setZeroes(matrix [][]int) {
	rows := make([]bool, len(matrix))
	cols := make([]bool, len(matrix[0]))
	for i, v := range matrix {
		for j, va := range v {
			if va == 0 {
				rows[i] = true
				cols[j] = true
			}
		}
	}

	for i, v := range matrix {
		for j := range v {
			if rows[i] || cols[j] {
				matrix[i][j] = 0
			}
		}
	}
}

// 最小路径和
func minPathSum(grid [][]int) int {
	if len(grid) == 0 || len(grid[0]) == 0 {
		return 0
	}

	dp := make([][]int, len(grid))
	for i := 0; i < len(dp); i++ {
		dp[i] = make([]int, len(grid[0]))
	}

	dp[0][0] = grid[0][0]
	for i := 1; i < len(grid); i++ {
		dp[i][0] = dp[i-1][0] + grid[i][0]
	}

	for j := 1; j < len(grid[0]); j++ {
		dp[0][j] = dp[0][j-1] + grid[0][j]
	}

	min := func(x, y int) int {
		if x > y {
			return y
		}

		return x
	}

	for i := 1; i < len(grid); i++ {
		for j := 1; j < len(grid[0]); j++ {
			dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j]
		}
	}

	return dp[len(grid)-1][len(grid[0])-1]
}

// 单词搜索
func wordExist(board [][]byte, word string) bool {
	if len(board) == 0 || len(board[0]) == 0 {
		return false
	}

	visited := make([][]bool, len(board))
	for i := range visited {
		visited[i] = make([]bool, len(board[0]))
	}

	var doCheck func(i, j, k int) bool
	doCheck = func(i, j, k int) bool {
		if i < 0 || j < 0 || i >= len(board) || j >= len(board[0]) || board[i][j] != word[k] || visited[i][j] {
			return false
		}

		if k == len(word)-1 {
			return true
		}

		visited[i][j] = true
		defer func() { visited[i][j] = false }()
		if doCheck(i-1, j, k+1) || doCheck(i+1, j, k+1) || doCheck(i, j-1, k+1) || doCheck(i, j+1, k+1) {
			return true
		}

		return false
	}

	for i := 0; i < len(board); i++ {
		for j := 0; j < len(board[0]); j++ {
			if doCheck(i, j, 0) {
				return true
			}
		}
	}

	return false
}

// 加油站加油
func canCompleteCircuit(gas []int, cost []int) int {
	var start, totalGas, curruntGas int
	for i := 0; i < len(gas); i++ {
		totalGas += gas[i] - cost[i]
		curruntGas += gas[i] - cost[i]
		if curruntGas < 0 {
			start = i + 1
			curruntGas = 0
		}
	}

	if totalGas < 0 {
		return -1
	}

	return start
}

// 解码方法
func numDecode(s string) int {
	dp := make([]int, len(s)+1)
	dp[0] = 1
	for i := 1; i <= len(s); i++ {
		if s[i-1] != '0' {
			dp[i] += dp[i-1]
		}

		if i > 1 && s[i-2] != '0' && (s[i-2]-'0')*10+(s[i-1]-'0') <= 26 {
			dp[i] += dp[i-2]
		}
	}

	return dp[len(s)]
}

func isRegxMatch(s string, p string) bool {
	if s < "a" || s > "z" || p < "a" || p > "z" {
		return false
	}

	return true
}

// LRU缓存机制
type LRUCache struct {
	capacity int
	cache    map[int]*list.Element
	list     *list.List
}

type kvValue struct {
	key   int
	value int
}

func NewLRUCache(capacity int) *LRUCache {
	return &LRUCache{
		capacity: capacity,
		cache:    make(map[int]*list.Element),
		list:     list.New(),
	}
}

func (lruCache *LRUCache) getFromLRUCache(key int) int {
	if element, ok := lruCache.cache[key]; ok {
		lruCache.list.MoveToFront(element)
		return element.Value.(*kvValue).value
	}

	return -1
}

func (lruCache *LRUCache) putToLRUCache(key int, value int) {
	if element, ok := lruCache.cache[key]; ok {
		lruCache.list.MoveToFront(element)
		element.Value.(*kvValue).value = value
	}

	element := lruCache.list.PushFront(&kvValue{key: key, value: value})
	lruCache.cache[key] = element
	if lruCache.list.Len() > lruCache.capacity {
		elem := lruCache.list.Back()
		if elem != nil {
			lruCache.list.Remove(elem)
			delete(lruCache.cache, elem.Value.(*kvValue).key)
		}
	}
}

// LFU缓存机制
type LFUCache struct {
	capacity     int
	len          int
	minFreq      int
	cache        map[int]*list.Element
	frequencyMap map[int]*list.List
}

type kvFrequency struct {
	key   int
	value int
	freq  int
}

func NewLFUCache(capacity int) *LFUCache {
	return &LFUCache{
		capacity:     capacity,
		len:          0,
		minFreq:      0,
		cache:        make(map[int]*list.Element),
		frequencyMap: make(map[int]*list.List),
	}
}

func (lfuCache *LFUCache) getFromLFUCache(key int) int {
	if element, ok := lfuCache.cache[key]; ok {
		lfuCache.increaseFrequency(element)
		return element.Value.(*kvFrequency).value
	}

	return -1
}

func (lfuCache *LFUCache) putToLFUCache(key int, value int) {
	if elem, ok := lfuCache.cache[key]; ok {
		kvFreq := elem.Value.(*kvFrequency)
		kvFreq.value = value
		lfuCache.increaseFrequency(elem)
		return
	}

	if lfuCache.len == lfuCache.capacity {
		lis := lfuCache.frequencyMap[lfuCache.minFreq]
		elem := lis.Back()
		kvFreq := elem.Value.(*kvFrequency)
		lis.Remove(elem)
		delete(lfuCache.cache, kvFreq.key)
		lfuCache.len--
	}

	lfuCache.insertToNewMap(&kvFrequency{key: key, value: value, freq: 1})
	lfuCache.minFreq = 1
	lfuCache.len++
}

func (lfuCache *LFUCache) increaseFrequency(elem *list.Element) {
	kvFreq := elem.Value.(*kvFrequency)
	oldest := lfuCache.frequencyMap[kvFreq.freq]
	oldest.Remove(elem)
	if lfuCache.minFreq == kvFreq.freq && oldest.Len() == 0 {
		lfuCache.minFreq++
	}

	lfuCache.insertToNewMap(kvFreq)

}

func (lfuCache *LFUCache) insertToNewMap(kvFreq *kvFrequency) {
	lis, ok := lfuCache.frequencyMap[kvFreq.freq]
	if !ok {
		lis = list.New()
		lfuCache.frequencyMap[kvFreq.freq] = lis
	}

	elem := lis.PushFront(kvFreq)
	lfuCache.cache[kvFreq.key] = elem
}

// 合并两个有序数组
func merge(nums1 []int, m int, nums2 []int, n int) {
	if m == 0 {
		for i, num := range nums2 {
			nums1[i] = num
		}

		return
	}

	if n == 0 {
		return
	}

	midNums := make([]int, len(nums1))
	for i, num := range nums1 {
		midNums[i] = num
	}

	index := 0
	i := 0
	j := 0
	for {
		if i == m || j == n {
			break
		}

		if midNums[i] < nums2[j] {
			nums1[index] = midNums[i]
			i++
			index++
		} else {
			nums1[index] = nums2[j]
			j++
			index++
		}
	}

	if i == m {
		for _, x := range nums2[j:] {
			nums1[index] = x
			index++
		}
	}

	if j == n {
		for _, x := range midNums[i:] {
			if x == 0 {
				continue
			}

			nums1[index] = x
			index++
		}
	}

	return
}

func removeElement(nums []int, val int) int {
	numMap := map[int]int{}
	for _, i := range nums {
		if i == val {
			continue
		}

		if _, ok := numMap[i]; ok {
			numMap[i]++
		} else {
			numMap[i] = 1
		}
	}

	var arr []int
	for k, v := range numMap {
		for i := 0; i < v; i++ {
			fmt.Printf("numMap k:%d, v:%d\n", k, v)
			arr = append(arr, k)
		}
	}

	for i := 0; i < len(arr); i++ {
		nums[i] = arr[i]
	}

	return len(arr)
}

func removeDuplicates(nums []int) int {
	minderMap := map[int]bool{}
	var s []int

	for i := 0; i < len(nums); i++ {
		if _, ok := minderMap[nums[i]]; ok {
			continue
		}

		minderMap[nums[i]] = true
		s = append(s, nums[i])
	}

	for i := 0; i < len(s); i++ {
		nums[i] = s[i]
	}

	return len(s)
}

func removeDuplicatesV2(nums []int) int {
	midMap := map[int]int{}
	var s []int

	for i := 0; i < len(nums); i++ {
		if _, ok := midMap[nums[i]]; ok {
			midMap[nums[i]]++
		} else {
			midMap[nums[i]] = 1
		}

		if x := midMap[nums[i]]; x <= 2 {
			s = append(s, nums[i])
		}
	}

	for i := 0; i < len(s); i++ {
		nums[i] = s[i]
	}

	return len(s)
}

func majorityElement(nums []int) int {
	if len(nums) == 0 {
		return 0
	}

	candidate := -1
	count := 0
	for i := 0; i < len(nums); i++ {
		if count == 0 {
			candidate = nums[i]
		}

		if nums[i] == candidate {
			count++
		} else {
			count--
		}
	}

	return candidate
}

/*
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
*/
/*func rotate(nums []int, k int) {
	k = k % len(nums)
	slices.Reverse(nums)
	slices.Reverse(nums[k:])
	slices.Reverse(nums[:k])
}

func maxProfit(prices []int) int {
	if len(prices) == 0 || len(prices) == 1 {
		return 0
	}

	maxPro := 0
	minCost := prices[0]

	minNum := func(x, y int) int {
		if x < y {
			return x
		}

		return y
	}

	maxNum := func(x, y int) int {
		if x > y {
			return x
		}

		return y
	}

	for i := 1; i < len(prices); i++ {
		maxPro = maxNum(maxPro, prices[i]-minCost)
		minCost = minNum(minCost, prices[i])
	}

	return maxPro
}

func hIndex(citations []int) int {
	if len(citations) == 0 {
		return 0
	}

	sort.Ints(citations)

	var h int
	for i := len(citations) - 1; i >= 0 && citations[i] >= h; i-- {
		h++
	}

	return h
}

type RandomizedSet struct {
	set   map[int]int
	array []int
}

func Constructor() RandomizedSet {
	return RandomizedSet{
		set:   make(map[int]int),
		array: []int{},
	}
}

func (this *RandomizedSet) Insert(val int) bool {
	if _, ok := this.set[val]; ok {
		return false
	}

	this.array = append(this.array, val)
	this.set[val] = len(this.array) - 1
	return true
}

func (this *RandomizedSet) Remove(val int) bool {
	value, ok := this.set[val]
	if !ok {
		return false
	}

	this.array[value] = this.array[len(this.array)-1]
	this.set[this.array[value]] = value
	this.array = this.array[:len(this.array)-1]
	delete(this.set, val)
	return true
}

func (this *RandomizedSet) GetRandom() int {
	return this.array[rand.Intn(len(this.array))]
}

func productExceptSelf(nums []int) []int {
	left := make([]int, len(nums))
	right := make([]int, len(nums))
	answer := make([]int, len(nums))

	left[0] = 1
	for i := 1; i < len(nums); i++ {
		left[i] = nums[i-1] * left[i-1]
	}

	right[len(nums)-1] = 1
	for i := len(nums) - 2; i >= 0; i-- {
		right[i] = nums[i+1] * right[i+1]
	}

	for i := 0; i < len(nums); i++ {
		answer[i] = left[i] * right[i]
	}

	return answer
}

func isPalindrome(s string) bool {
	str := ""
	for i := 0; i < len(s); i++ {
		if s[i] >= 48 && s[i] <= 57 || s[i] >= 97 && s[i] <= 122 {
			str += string(s[i])
		}

		if s[i] >= 65 && s[i] <= 90 {
			str += string(s[i] + 32)
		}
	}

	for i := 0; i < len(str)/2; i++ {
		if str[i] != str[len(str)-i-1] {
			return false
		}
	}

	return true
}

func isSubsequence(s string, t string) bool {
	if len(s) == 0 {
		return true
	}

	if len(t) == 0 {
		return false
	}

	sIndex, tIndex := 0, 0
	for ; tIndex < len(t) && sIndex < len(s); tIndex++ {
		if t[tIndex] == s[sIndex] {
			sIndex++
		}
	}

	if sIndex == len(s) {
		return true
	}

	return false
}

func twoSum(numbers []int, target int) []int {
	i, j := 0, len(numbers)-1
	for i < j {
		if numbers[i]+numbers[j] < target {
			i++
			continue
		}

		if numbers[i]+numbers[j] < target {
			j--
			continue
		}

		return []int{i, j}
	}

	return []int{-1, -1}
}

func maxArea(height []int) int {
	i, j := 0, len(height)-1

	maxSize := min(height[i], height[j]) * (j - i)
	for i < j {
		if height[i] < height[j] {
			i++
		} else {
			j--
		}

		size := min(height[i], height[j]) * (j - i)
		if size > maxSize {
			maxSize = size
		}
	}

	return maxSize
}

func threeSum(nums []int) [][]int {
	var answer [][]int
	sort.Ints(nums)
	for i := 0; i < len(nums); i++ {
		if i > 0 && nums[i] == nums[i-1] {
			continue
		}

		k := len(nums) - 1
		for j := i + 1; j < len(nums); j++ {
			if j > i+1 && nums[j] == nums[j-1] {
				continue
			}

			target := (-1) * nums[i]
			for j < k && nums[j]+nums[k] > target {
				k--
			}

			if j == k {
				break
			}

			if nums[j]+nums[k] == target {
				answer = append(answer, []int{nums[i], nums[j], nums[k]})
			}
		}
	}

	return answer
}*/

func minSubArrayLen(target int, nums []int) int {
	if len(nums) == 0 {
		return 0
	}

	i, j, sum, ans := 0, 0, 0, math.MaxInt
	for j < len(nums) {
		sum += nums[j]
		for sum >= target {
			ans = min(ans, j-i+1)
			sum -= nums[i]
			i++
		}

		j++
	}

	if ans == math.MaxInt {
		ans = 0
	}

	return ans
}

func lengthOfLongestSubstring(s string) int {
	strMap := map[string]int{}

	i, j, ans := 0, 0, 0
	for j < len(s) {
		if index, ok := strMap[string(s[j])]; ok && index >= i {
			i = index + 1
		}

		strMap[string(s[j])] = j
		ans = max(ans, j-i+1)
		j++
	}

	return ans
}

func isValidSudoku(board [][]byte) bool {
	var row, col, box [9][10]int
	for i := 0; i < 9; i++ {
		for j := 0; j < 9; j++ {
			if board[i][j] == '.' {
				continue
			}

			num := board[i][j] - '0'
			if row[i][num] == 1 || col[j][num] == 1 || box[j/3+(i/3)*3][num] == 1 {
				return false
			}

			row[i][num] = 1
			col[j][num] = 1
			box[j/3+(i/3)*3][num] = 1
		}
	}

	return true
}

func spiralOrder(matrix [][]int) []int {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return []int{}
	}

	row, col := len(matrix), len(matrix[0])
	ans := make([]int, row*col)
	top, bottom, left, right, index := 0, row-1, 0, col-1, 0 //上下左右边界

	for top <= bottom && left <= right {
		for j := left; j <= right; j++ {
			ans[index] = matrix[top][j]
			index++
		}

		for i := top + 1; i <= bottom; i++ {
			ans[index] = matrix[i][right]
			index++
		}

		if left < right && top < bottom {
			for j := right - 1; j > left; j-- {
				ans[index] = matrix[bottom][j]
				index++
			}

			for i := bottom; i > top; i-- {
				ans[index] = matrix[i][left]
				index++
			}
		}

		left++
		top++
		right--
		bottom--
	}

	return ans
}

func rotate(matrix [][]int) {
	n := len(matrix)
	if n == 0 {
		return
	}

	top, bottom, left, right := 0, n-1, 0, n-1
	for top < bottom && left < right {
		for i := left; i < right; i++ {
			a := matrix[top][i]
			b := matrix[i][n-top-1]
			c := matrix[n-top-1][n-i-1]
			d := matrix[n-i-1][top]

			matrix[top][i] = d
			matrix[i][n-top-1] = a
			matrix[n-top-1][n-i-1] = b
			matrix[n-i-1][top] = c
		}

		top++
		bottom--
		left++
		right--
	}

	return
}

func setZeroes(matrix [][]int) {
	if len(matrix) == 0 || len(matrix[0]) == 0 {
		return
	}

	rows := make([]bool, len(matrix))
	cols := make([]bool, len(matrix[0]))

	for i, r := range matrix {
		for j, c := range r {
			if c == 0 {
				rows[i] = true
				cols[j] = true
			}
		}
	}

	for i, row := range matrix {
		for j := range row {
			if rows[i] || cols[j] {
				matrix[i][j] = 0
			}
		}
	}

	return
}

func canConstruct(ransomNote string, magazine string) bool {
	mp := map[byte]int{}
	rp := map[byte]int{}

	for i := 0; i < len(magazine); i++ {
		if _, ok := mp[magazine[i]]; ok {
			mp[magazine[i]]++
		} else {
			mp[magazine[i]] = 1
		}
	}

	for i := 0; i < len(ransomNote); i++ {
		if _, ok := rp[ransomNote[i]]; ok {
			rp[ransomNote[i]]++
		} else {
			rp[ransomNote[i]] = 1
		}
	}

	for key, value := range rp {
		if count, ok := mp[key]; !ok || count < value {
			return false
		}
	}

	return true
}

func isIsomorphic(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}

	s2tMap := map[byte]byte{}
	t2sMap := map[byte]byte{}
	for i := 0; i < len(s); i++ {
		if s2tMap[s[i]] > 0 && s2tMap[s[i]] != t[i] || t2sMap[t[i]] > 0 && t2sMap[t[i]] != s[i] {
			return false
		}

		s2tMap[s[i]] = t[i]
		t2sMap[t[i]] = s[i]
	}

	return true
}

func wordPattern(pattern string, s string) bool {
	strSli := strings.Split(s, " ")
	if len(pattern) != len(strSli) {
		return false
	}

	p2sMap := map[byte]string{}
	s2Map := map[string]byte{}

	for i := range pattern {
		pa := pattern[i]
		str := strSli[i]
		if p2sMap[pa] != "" && p2sMap[pa] != str || s2Map[str] > 0 && s2Map[str] != pa {
			return false
		}

		p2sMap[pa] = str
		s2Map[str] = pa
	}

	return true
}

func isAnagram(s string, t string) bool {
	if len(s) != len(t) {
		return false
	}

	mp := map[rune]int{}
	for i := 0; i < len(s); i++ {
		if _, ok := mp[int32(s[i])]; ok {
			mp[int32(s[i])]++
		} else {
			mp[int32(s[i])] = 1
		}
	}

	for i := 0; i < len(t); i++ {
		value, ok := mp[int32(t[i])]
		if !ok {
			return false
		}

		if value == 0 {
			return false
		}

		mp[int32(t[i])]--
	}

	return true
}

func hasCycle(head *ListNode) bool {
	if head == nil || head.Next == nil {
		return false
	}

	fast := head.Next
	slow := head

	for fast != slow {
		if fast == nil || slow == nil {
			return false
		}

		fast = fast.Next.Next
		slow = slow.Next
	}

	return true
}

func addTwoNumbers(l1 *ListNode, l2 *ListNode) *ListNode {
	var ans, tail *ListNode

	count := 0
	for l1 != nil || l2 != nil {
		sum := 0
		if l1 != nil {
			sum += l1.Val
			l1 = l1.Next
		}

		if l2 != nil {
			sum += l2.Val
			l2 = l2.Next
		}

		sum += count
		count = sum / 10
		if ans == nil {
			ans = &ListNode{
				Val: sum % 10,
			}

			tail = ans
		} else {
			tail.Next = &ListNode{
				Val: sum % 10,
			}

			tail = tail.Next
		}
	}

	if count != 0 {
		tail.Next = &ListNode{Val: count}
	}

	return ans
}

func mergeTwoLists(list1 *ListNode, list2 *ListNode) *ListNode {
	ans := &ListNode{}
	tail := ans

	for list1 != nil && list2 != nil {
		if list1.Val < list2.Val {
			tail.Next = list1
			list1 = list1.Next
		} else {
			tail.Next = list2
			list2 = list2.Next
		}

		tail = tail.Next
	}

	if list1 != nil {
		tail.Next = list1
	}

	if list2 != nil {
		tail.Next = list2
	}

	return ans.Next
}

func copyRandomList(head *Node) *Node {
	if head == nil {
		return nil
	}

	oldToNew := make(map[*Node]*Node)
	tail := head

	for tail != nil {
		oldToNew[tail] = &Node{Val: tail.Val}
		tail = tail.Next
	}

	tail = head
	for tail != nil {
		oldToNew[tail].Next = oldToNew[tail.Next]
		oldToNew[tail].Random = oldToNew[tail.Random]
		tail = tail.Next
	}

	return oldToNew[head]
}

func romanToInt(s string) int {
	m := map[uint8]int{
		'I': 1,
		'V': 5,
		'X': 10,
		'L': 50,
		'C': 100,
		'D': 500,
		'M': 1000,
	}

	mp := map[string]int{
		"IV": 4,
		"IX": 9,
		"XL": 40,
		"XC": 90,
		"CD": 400,
		"CM": 900,
	}

	if len(s) == 0 {
		return 0
	}

	var ans int
	i := 0
	for ; i < len(s)-1; i++ {
		if s[i] == 'I' || s[i] == 'X' || s[i] == 'C' {
			s2 := string(s[i]) + string(s[i+1])
			if value, ok := mp[s2]; ok {
				ans += value
				i++
				continue
			}
		}

		value, _ := m[s[i]]
		ans += value
	}

	if i == len(s)-1 {
		value, _ := m[s[i]]
		ans += value
	}

	return ans
}

func intToRoman(num int) string {
	ans := ""
	mp := map[int]string{
		1:    "I",
		4:    "IV",
		5:    "V",
		9:    "IX",
		10:   "X",
		40:   "XL",
		50:   "L",
		90:   "XC",
		100:  "C",
		400:  "CD",
		500:  "D",
		900:  "CM",
		1000: "M",
	}

	sli := []int{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}
	for i := range sli {
		for num >= sli[i] {
			num -= sli[i]
			s, _ := mp[sli[i]]
			ans += s
		}

		if num == 0 {
			break
		}
	}

	return ans
}

func lengthOfLastWord(s string) int {
	strSli := strings.Split(s, " ")
	var str []string
	for i := range strSli {
		if strSli[i] == "" {
			continue
		}

		str = append(str, strSli[i])
	}

	return len(str[len(str)-1])
}

func longestCommonPrefix(strs []string) string {
	if len(strs) == 0 || len(strs[0]) == 0 {
		return ""
	}

	ans := ""
	str := ""
	s := strs[0]
	for i := range s {
		str += string(s[i])

		for j := 1; j < len(strs); j++ {
			if !strings.HasPrefix(strs[j], str) {
				return ans
			}
		}

		ans = str
	}

	return ans
}

func reverseWords(s string) string {
	i := 0
	for string(s[i]) == " " {
		i++
	}

	var strs []string
	for i < len(s) {
		str := ""
		for i < len(s) && string(s[i]) != " " {
			str += string(s[i])
			i++
		}

		strs = append(strs, str)
		for i < len(s) && string(s[i]) == " " {
			i++
		}
	}

	ans := ""
	for j := len(strs) - 1; j >= 0; j-- {
		ans += string(strs[j])
		if j != 0 {
			ans += " "
		}
	}

	return ans
}

func convert(s string, numRows int) string {
	if numRows < 2 {
		return s
	}

	str := make([]string, numRows)
	i, flag := 0, -1
	for j := range s {
		str[i] += string(s[j])
		if i == 0 || i == numRows-1 {
			flag = -flag
		}

		i += flag
	}

	ans := ""

	for j := range str {
		ans += string(str[j])
	}

	return ans
}

func strStr(haystack string, needle string) int {
	return strings.Index(haystack, needle)
}

func isHappy(n int) bool {
	slow, fast := n, calcNextNum(n)
	for fast != 1 && slow != fast {
		slow = calcNextNum(slow)
		fast = calcNextNum(calcNextNum(fast))
	}

	return fast == 1
}

func calcNextNum(n int) int {
	nextNum := 0
	for n > 0 {
		nextNum += (n % 10) * (n % 10)
		n /= 10
	}

	return nextNum
}

func containsNearbyDuplicate(nums []int, k int) bool {
	if len(nums) < 2 {
		return false
	}

	mp := map[int]int{}
	for index := range nums {
		if value, ok := mp[nums[index]]; ok {
			if index-value <= k {
				return true
			}
		}

		mp[nums[index]] = index
	}

	return false
}

func summaryRanges(nums []int) []string {
	i, j := 0, 0
	var ans []string
	for j < len(nums)-1 {
		for j < len(nums)-1 && nums[j]+1 == nums[j+1] {
			j++
		}

		if i == j {
			ans = append(ans, strconv.Itoa(nums[i]))
		} else {
			ans = append(ans, strconv.Itoa(nums[i])+"->"+strconv.Itoa(nums[j]))
		}

		j++
		i = j
	}

	if i == j && i < len(nums) {
		ans = append(ans, strconv.Itoa(nums[i]))
	}

	return ans
}

func merge(intervals [][]int) [][]int {
	if len(intervals) == 0 || len(intervals[0]) == 0 {
		return [][]int{}
	}

	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})

	var ans [][]int
	for _, in := range intervals {
		l := len(ans)
		if l > 0 && ans[l-1][1] >= in[0] {
			ans[l-1][1] = max(in[1], ans[l-1][1])
		} else {
			ans = append(ans, in)
		}
	}

	return ans
}

func insert(intervals [][]int, newInterval []int) [][]int {
	if len(intervals) == 0 || len(intervals[0]) == 0 {
		return [][]int{}
	}

	intervals = append(intervals, newInterval)
	sort.Slice(intervals, func(i, j int) bool {
		return intervals[i][0] < intervals[j][0]
	})

	var ans [][]int
	ans = append(ans, intervals[0])
	for _, in := range intervals[1:] {
		if ans[len(ans)-1][1] < in[0] {
			ans = append(ans, in)
		} else {
			ans[len(ans)-1][1] = max(ans[len(ans)-1][1], in[1])
		}
	}

	return ans
}

func findMinArrowShots(points [][]int) int {
	if len(points) == 0 || len(points[0]) == 0 {
		return 0
	}

	sort.Slice(points, func(i, j int) bool {
		return points[i][0] < points[j][0]
	})

	var ans [][]int
	ans = append(ans, points[0])
	for _, in := range points[1:] {
		if len(ans[len(ans)-1]) == 1 {
			if in[0] > ans[len(ans)-1][0] {
				ans = append(ans, in)
			}

			continue
		}

		if in[0] < ans[len(ans)-1][1] {
			ans[len(ans)-1][0] = in[0]
			ans[len(ans)-1][1] = min(ans[len(ans)-1][1], in[1])
		} else if in[0] > ans[len(ans)-1][1] {
			ans = append(ans, in)
		} else {
			ans = ans[:len(ans)-1]
			ans = append(ans, []int{in[0]})
		}
	}

	return len(ans)
}

func isValid(s string) bool {
	var strs []uint8
	mp := map[uint8]uint8{
		')': '(',
		']': '[',
		'}': '{',
	}

	for i := 0; i < len(s); i++ {
		if s[i] == '(' || s[i] == '[' || s[i] == '{' {
			strs = append(strs, s[i])
		} else {
			if len(strs) == 0 {
				return false
			}

			if value, _ := mp[s[i]]; value != strs[len(strs)-1] {
				return false
			}

			strs = strs[:len(strs)-1]
		}
	}

	if len(strs) != 0 {
		return false
	}

	return true
}

func simplifyPath(path string) string {
	if len(path) == 0 {
		return ""
	}

	var ans []string
	strs := strings.Split(path, "/")
	for _, s := range strs {
		if s == ".." {
			if len(ans) > 0 {
				ans = ans[:len(ans)-1]
			}

		} else if s != "" && s != "." {
			ans = append(ans, s)
		}
	}

	return "/" + strings.Join(ans, "/")
}

type MinStack struct {
	index int
	num   []int
}

func ConstructorMinStack() MinStack {
	return MinStack{
		index: -1,
		num:   make([]int, 0),
	}
}

func (this *MinStack) Push(val int) {
	if len(this.num) == 0 {
		this.index = 0
	} else {
		if val < this.num[this.index] {
			this.index = len(this.num)
		}
	}

	this.num = append(this.num, val)
}

func (this *MinStack) Pop() {
	this.num = this.num[:len(this.num)-1]
	if this.index == len(this.num) {
		minNum := math.MaxInt
		for index, n := range this.num {
			if n < minNum {
				this.index = index
				minNum = n
			}
		}
	}
}

func (this *MinStack) Top() int {
	return this.num[len(this.num)-1]
}

func (this *MinStack) GetMin() int {
	return this.num[this.index]
}

func evalRPN(tokens []string) int {
	var num []int
	for _, t := range tokens {
		if t != "+" && t != "-" && t != "*" && t != "/" {
			n, _ := strconv.Atoi(t)
			num = append(num, n)
			continue
		}

		firstNum := num[len(num)-2]
		secondNum := num[len(num)-1]
		num = num[:len(num)-2]
		switch t {
		case "+":
			num = append(num, firstNum+secondNum)
		case "-":
			num = append(num, firstNum-secondNum)
		case "*":
			num = append(num, firstNum*secondNum)
		case "/":
			num = append(num, firstNum/secondNum)
		}
	}

	return num[0]
}

// todo
/*func calculate(s string) int {
	var numStack []int
	var strStack []string
	for _, s1 := range s {
		t := string(s1)
		if t == " " {
			continue
		}

		if t != "+" && t != "-" && t != "*" && t != "/" && t != "(" && t != ")" {
			n, _ := strconv.Atoi(t)
			numStack = append(numStack, n)
			continue
		}

		if t == ")" {
			for len(strStack) > 0 && strStack[len(strStack)-1] != "(" {
				str := strStack[len(strStack)-1]
				ans := calc(str, numStack[len(numStack)-2], numStack[len(numStack)-1])
				numStack = numStack[:len(numStack)-2]
				numStack = append(numStack, ans)
				strStack = strStack[:len(strStack)-1]
			}

			strStack = strStack[:len(strStack)-1]
			continue
		}

		strStack = append(strStack, t)
	}

	for len(strStack) > 0 {
		str := strStack[0]
		var ans int
		if str == "-" && len(numStack) == 1 {
			ans = 0 - numStack[0]
			numStack[0] = ans
		} else {
			ans = calc(str, numStack[0], numStack[1])
			numStack = numStack[2:]
			strStack = strStack[1:]
		}

		numStack = append(numStack, ans)
		strStack = strStack[:len(numStack)-1]
	}

	return numStack[0]
}

func calc(t string, firstNum, secondNum int) int {
	switch t {
	case "+":
		return firstNum + secondNum
	case "-":
		return firstNum - secondNum
	case "*":
		return firstNum * secondNum
	case "/":
		return firstNum / secondNum
	}

	return 0
}
*/

func reverseBetween(head *ListNode, left int, right int) *ListNode {
	if head == nil {
		return nil
	}

	dummyNode := &ListNode{}
	dummyNode.Next = head
	i := 1
	tail := dummyNode
	for ; i < left; i++ {
		tail = tail.Next
	}

	n1 := tail
	n2 := tail.Next

	for ; i <= right; i++ {
		tail = tail.Next
	}

	n3 := tail
	n4 := tail.Next

	n1.Next = nil
	n3.Next = nil

	reverseLinkedList(n2)
	n1.Next = n3
	n2.Next = n4

	return head
}

func reverseLinkedList(head *ListNode) {
	var pre *ListNode
	tail := head
	for tail != nil {
		next := tail.Next
		tail.Next = pre
		pre = tail
		tail = next
	}
}

func removeNthFromEnd(head *ListNode, n int) *ListNode {
	dummyNode := &ListNode{}
	dummyNode.Next = head
	tail := dummyNode

	cur := head
	length := 0
	for cur != nil {
		length++
		cur = cur.Next
	}

	for i := 0; i < length-n; i++ {
		tail = tail.Next
	}

	tail.Next = tail.Next.Next
	return dummyNode.Next
}

func deleteDuplicates(head *ListNode) *ListNode {
	if head == nil {
		return nil
	}

	dummpyNode := &ListNode{}
	dummpyNode.Next = head
	tail := dummpyNode

	for tail.Next != nil && tail.Next.Next != nil {
		if tail.Next.Val == tail.Next.Next.Val {
			num := tail.Next.Val
			for tail.Next != nil && tail.Next.Val == num {
				tail.Next = tail.Next.Next
			}
		} else {
			tail = tail.Next
		}
	}

	return dummpyNode.Next
}

func rotateRight(head *ListNode, k int) *ListNode {
	if head == nil {
		return nil
	}

	tail := head
	listLen := 0
	for tail != nil {
		listLen++
		tail = tail.Next
	}

	k = k % listLen
	if k == 0 {
		return head
	}

	dummyNode := &ListNode{}
	dummyNode.Next = head
	tail = dummyNode
	for i := 0; i < listLen-k; i++ {
		tail = tail.Next
	}

	next := tail.Next
	tail.Next = nil
	dummyNode.Next = next
	tail = dummyNode
	for tail.Next != nil {
		tail = tail.Next
	}

	tail.Next = head

	return dummyNode.Next
}

func getPartition(head *ListNode, x int) *ListNode {
	if head == nil {
		return nil
	}

	dummyNode := &ListNode{}
	minList, maxList := &ListNode{}, &ListNode{}
	m1, m2 := minList, maxList
	tail := head
	for tail != nil {
		if tail.Val < x {
			minList.Val = tail.Val
			minList.Next = &ListNode{}
			minList = minList.Next
		} else {
			maxList.Val = tail.Val
			maxList.Next = &ListNode{}
			maxList = maxList.Next
		}

		tail = tail.Next
	}

	if m1.Next == nil {
		dummyNode.Next = m2
	} else {
		dummyNode.Next = m1
		tail = dummyNode.Next
		for tail.Next.Next != nil {
			tail = tail.Next
		}

		tail.Next = m2
	}

	tail = dummyNode.Next
	for tail.Next.Next != nil {
		tail = tail.Next
	}

	tail.Next = nil
	return dummyNode.Next
}

type LRUCache struct {
	capacity int
	size     int
	cache    map[int]*TwoWayListNode
	head     *TwoWayListNode
	tail     *TwoWayListNode
}

func Constructor(capacity int) LRUCache {
	l := LRUCache{
		capacity: capacity,
		size:     0,
		cache:    make(map[int]*TwoWayListNode),
		head:     &TwoWayListNode{},
		tail:     &TwoWayListNode{},
	}

	l.head.Next = l.tail
	l.tail.Pre = l.head
	return l
}

func (l *LRUCache) Get(key int) int {
	if v, ok := l.cache[key]; ok {
		l.moveToHead(v)
		return v.Value
	}

	return -1
}

func (l *LRUCache) Put(key int, value int) {
	if v, ok := l.cache[key]; ok {
		v.Value = value
		l.moveToHead(v)
	}

	node := &TwoWayListNode{
		Key:   key,
		Value: value,
	}

	l.cache[key] = node
	l.addNodeToHead(node)
	l.size++
	if l.size > l.capacity {
		removedKey := l.removeNodeFromTail()
		delete(l.cache, removedKey)
		l.size--
	}
}

func (l *LRUCache) addNodeToHead(node *TwoWayListNode) {
	node.Pre = l.head
	node.Next = l.head.Next
	l.head.Next.Pre = node
	l.head.Next = node
}

func (l *LRUCache) removeNode(node *TwoWayListNode) {
	node.Pre.Next = node.Next
	node.Next.Pre = node.Pre
}

func (l *LRUCache) moveToHead(node *TwoWayListNode) {
	l.removeNode(node)
	l.addNodeToHead(node)
}

func (l *LRUCache) removeNodeFromTail() int {
	node := l.tail.Pre
	l.removeNode(node)
	return node.Key
}

func maxDepthV1(root *TreeNode) int {
	if root == nil {
		return 0
	}

	return max(maxDepth(root.Left), maxDepth(root.Right)) + 1
}

func maxDepth(root *TreeNode) int {
	if root == nil {
		return 0
	}

	var queue []*TreeNode
	queue = append(queue, root)
	ans := 0
	for len(queue) > 0 {
		length := len(queue)
		for length > 0 {
			node := queue[0]
			queue = queue[1:]
			if node.Left != nil {
				queue = append(queue, node.Left)
			}

			if node.Right != nil {
				queue = append(queue, node.Right)
			}

			length--
		}

		ans++
	}

	return ans
}

func isSameTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}

	if p == nil || q == nil {
		return false
	}

	var pQueue []*TreeNode
	var qQueue []*TreeNode
	pQueue = append(pQueue, p)
	qQueue = append(qQueue, q)
	for len(pQueue) > 0 && len(qQueue) > 0 {
		node1, node2 := pQueue[0], qQueue[0]
		pQueue, qQueue = pQueue[1:], qQueue[1:]
		if node1.Val != node2.Val {
			return false
		}

		l1, r1 := node1.Left, node1.Right
		l2, r2 := node2.Left, node2.Right
		if l1 != nil && l2 == nil || l1 == nil && l2 != nil {
			return false
		}

		if r1 != nil && r2 == nil || r1 == nil && r2 != nil {
			return false
		}

		if l1 != nil {
			pQueue = append(pQueue, l1)
		}

		if r1 != nil {
			pQueue = append(pQueue, r1)
		}

		if l2 != nil {
			qQueue = append(qQueue, l2)
		}

		if r2 != nil {
			qQueue = append(qQueue, r2)
		}
	}

	return len(pQueue) == 0 && len(qQueue) == 0
}

func invertTree(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}

	left := invertTree(root.Left)
	right := invertTree(root.Right)
	root.Left = left
	root.Right = right
	return root
}

func isSymmetric(root *TreeNode) bool {
	//return checkTree(root, root)
	var queue []*TreeNode
	queue = append(queue, []*TreeNode{root, root}...)
	for len(queue) > 0 {
		node0, node1 := queue[0], queue[1]
		queue = queue[2:]
		if node0 == nil && node1 == nil {
			continue
		}

		if node0 == nil || node1 == nil {
			return false
		}

		if node0.Val != node1.Val {
			return false
		}

		queue = append(queue, node0.Left)
		queue = append(queue, node1.Right)
		queue = append(queue, node0.Right)
		queue = append(queue, node1.Left)
	}

	return true
}

func checkTree(p *TreeNode, q *TreeNode) bool {
	if p == nil && q == nil {
		return true
	}

	if p == nil || q == nil {
		return false
	}

	if p.Val != q.Val {
		return false
	}

	return checkTree(p.Left, q.Right) && checkTree(p.Right, q.Left)
}

func buildTreePre(preorder []int, inorder []int) *TreeNode {
	if len(preorder) == 0 || len(inorder) == 0 {
		return nil
	}

	rootVal := preorder[0]
	root := &TreeNode{
		Val: rootVal,
	}

	i := 0
	for ; i < len(inorder); i++ {
		if inorder[i] == rootVal {
			break
		}
	}

	root.Left = buildTreePre(preorder[1:len(inorder[:i])+1], inorder[:i])
	root.Right = buildTreePre(preorder[len(inorder[:i])+1:], inorder[i+1:])
	return root
}

func buildTree(inorder []int, postorder []int) *TreeNode {
	if len(inorder) == 0 || len(postorder) == 0 {
		return nil
	}

	rootVal := postorder[len(postorder)-1]
	root := &TreeNode{Val: rootVal}
	i := 0
	for ; i < len(inorder); i++ {
		if inorder[i] == rootVal {
			break
		}
	}

	root.Left = buildTree(inorder[:i], postorder[:i])
	root.Right = buildTree(inorder[i+1:], postorder[i:len(inorder)-1])
	return root
}

func connect(root *TreeNode) *TreeNode {
	if root == nil {
		return nil
	}

	var queue []*TreeNode
	queue = append(queue, root)
	for len(queue) > 0 {
		tmp := queue
		queue = nil
		for i, node := range tmp {
			if i+1 < len(tmp) {
				node.Next = tmp[i+1]
			}

			if node.Left != nil {
				queue = append(queue, node.Left)
			}

			if node.Right != nil {
				queue = append(queue, node.Right)
			}
		}
	}

	return root
}

func flatten(root *TreeNode) {
	if root == nil {
		return
	}

	list := preOrder(root)
	for i := 0; i < len(list)-1; i++ {
		cur, next := list[i], list[i+1]
		cur.Left, cur.Right = nil, next
	}
}

func preOrder(node *TreeNode) []*TreeNode {
	var list []*TreeNode
	if node == nil {
		return list
	}

	list = append(list, node)
	list = append(list, preOrder(node.Left)...)
	list = append(list, preOrder(node.Right)...)
	return list
}

func hasPathSum(root *TreeNode, targetSum int) bool {
	if root == nil {
		return false
	}

	if root.Val == targetSum && root.Left == nil && root.Right == nil {
		return true
	}

	return hasPathSum(root.Left, targetSum-root.Val) || hasPathSum(root.Right, targetSum-root.Val)
}

func sumNumbers(root *TreeNode) int {
	//return dfsSumNums(0, root)
	if root == nil {
		return 0
	}

	queue := []*TreeNode{root}
	nums := []int{root.Val}
	ans := 0
	for len(queue) > 0 {
		node := queue[0]
		queue = queue[1:]
		num := nums[0]
		nums = nums[1:]
		if node.Left == nil && node.Right == nil {
			ans += num
		}

		if node.Left != nil {
			queue = append(queue, node.Left)
			nums = append(nums, num*10+node.Left.Val)
		}

		if node.Right != nil {
			queue = append(queue, node.Right)
			nums = append(nums, num*10+node.Right.Val)
		}
	}

	return ans
}

func dfsSumNums(num int, root *TreeNode) int {
	if root == nil {
		return 0
	}

	num = num*10 + root.Val
	if root.Left == nil && root.Right == nil {
		return num
	}

	return dfsSumNums(num, root.Left) + dfsSumNums(num, root.Right)
}

var (
	path []byte
	ans  []string
	mp   map[byte]string
)

func letterCombinations(digits string) []string {
	if len(digits) == 0 {
		return []string{}
	}

	mp = map[byte]string{
		'2': "abc",
		'3': "def",
		'4': "ghi",
		'5': "jkl",
		'6': "mno",
		'7': "pqrs",
		'8': "tuv",
		'9': "wxyz",
	}

	path = make([]byte, 0)
	ans = make([]string, 0)
	dfsLetterCombinations(digits, 0)
	return ans
}

func dfsLetterCombinations(digits string, index int) {
	if len(path) == len(digits) {
		ans = append(ans, string(path))
		return
	}

	str, _ := mp[digits[index]]
	for i := 0; i < len(str); i++ {
		path = append(path, str[i])
		dfsLetterCombinations(digits, index+1)
		path = path[:len(path)-1]
	}
}

var (
	pathCombine []int
	ansCombine  [][]int
)

func combine(n int, k int) [][]int {
	pathCombine = make([]int, 0)
	ansCombine = make([][]int, 0)
	dfsCombine(n, k, 1)
	return ansCombine
}

func dfsCombine(n int, k int, index int) {
	if len(pathCombine) == k {
		tmp := make([]int, k)
		copy(tmp, pathCombine)
		ansCombine = append(ansCombine, tmp)
		return
	}

	for i := index; i <= n; i++ {
		pathCombine = append(pathCombine, i)
		dfsCombine(n, k, i+1)
		pathCombine = pathCombine[:len(pathCombine)-1]
	}
}

var (
	pathPermute []int
	ansPermute  [][]int
	used        []bool
)

func permute(nums []int) [][]int {
	pathPermute, ansPermute, used = make([]int, 0), make([][]int, 0), make([]bool, len(nums))
	dfsPermute(nums, 0)
	return ansPermute
}

func dfsPermute(nums []int, index int) {
	if index == len(nums) {
		tmp := make([]int, index)
		copy(tmp, pathPermute)
		ansPermute = append(ansPermute, tmp)
		return
	}

	for i := 0; i < len(nums); i++ {
		if used[i] {
			continue
		}

		pathPermute = append(pathPermute, nums[i])
		used[i] = true
		dfsPermute(nums, index+1)
		used[i] = false
		pathPermute = pathPermute[:len(pathPermute)-1]
	}
}

var (
	pathCombinationSum []int
	ansCombinationSum  [][]int
)

func combinationSum(candidates []int, target int) [][]int {
	pathCombinationSum, ansCombinationSum = make([]int, 0), make([][]int, 0)
	sort.Ints(candidates)
	dfsCombinationSum(candidates, 0, target)
	return ansCombinationSum
}

func dfsCombinationSum(candidates []int, index int, target int) {
	if target == 0 {
		tmp := make([]int, len(pathCombinationSum))
		copy(tmp, pathCombinationSum)
		ansCombinationSum = append(ansCombinationSum, tmp)
		return
	}

	for i := index; i < len(candidates); i++ {
		if candidates[i] > target {
			break
		}

		pathCombinationSum = append(pathCombinationSum, candidates[i])
		dfsCombinationSum(candidates, i, target-candidates[i])
		pathCombinationSum = pathCombinationSum[:len(pathCombinationSum)-1]
	}
}

type pair struct {
	x int
	y int
}

func exist(board [][]byte, word string) bool {
	visited := make([][]bool, len(board))
	diraction := []pair{{-1, 0}, {1, 0}, {0, -1}, {0, 1}}
	for i := range visited {
		visited[i] = make([]bool, len(board[0]))
	}

	var checkNext func(i, j, index int) bool
	checkNext = func(i, j, index int) bool {
		if board[i][j] != word[index] {
			return false
		}

		if index == len(word)-1 {
			return true
		}

		visited[i][j] = true
		for _, d := range diraction {
			nextI, nextJ := i+d.x, j+d.y
			if nextI >= 0 && nextI < len(board) && nextJ >= 0 && nextJ < len(board[0]) && !visited[nextI][nextJ] {
				if checkNext(nextI, nextJ, index+1) {
					return true
				}
			}
		}

		visited[i][j] = false
		return false
	}

	for i := range board {
		for j := range board[0] {
			if checkNext(i, j, 0) {
				return true
			}
		}
	}

	return false
}

func generateParenthesis(n int) []string {
	path := make([]byte, 2*n)
	var (
		ans []string
		dfs func(i, left int)
	)

	dfs = func(i, left int) {
		if i == n*2 {
			ans = append(ans, string(path))
			return
		}

		if left < n {
			path[i] = '('
			dfs(i+1, left+1)
		}

		if i-left < left {
			path[i] = ')'
			dfs(i+1, left)
		}
	}

	dfs(0, 0)
	return ans
}

// n皇后
func totalNQueens(n int) int {
	chess := make([][]string, n)
	var ans [][]string
	for i := range chess {
		c := make([]string, n)
		for j := 0; j < n; j++ {
			c = append(c, ".")
		}

		chess[i] = c
	}

	var checkValidate func(int, int, [][]string) bool
	checkValidate = func(i, j int, chess [][]string) bool {
		for x := 0; x < i; x++ {
			if chess[x][j] == "Q" {
				return false
			}
		}

		for x := 0; x < j; x++ {
			if chess[i][x] == "Q" {
				return false
			}
		}

		for x, y := i-1, j-1; x >= 0 && y >= 0; x, y = x-1, y-1 {
			if chess[x][y] == "Q" {
				return false
			}
		}

		for x, y := i-1, j+1; x >= 0 && y < n; x, y = x-1, y+1 {
			if chess[x][y] == "Q" {
				return false
			}
		}

		return true
	}

	var dfs func(int)
	dfs = func(row int) {
		if row == n {
			tmp := make([]string, n)
			for i, str := range chess {
				tmp[i] = strings.Join(str, "")
			}

			ans = append(ans, tmp)
		}

		for i := 0; i < n; i++ {
			if checkValidate(row, i, chess) {
				chess[row][i] = "Q"
				dfs(row + 1)
				chess[row][i] = "."
			}
		}
	}

	dfs(0)
	return len(ans)
}
