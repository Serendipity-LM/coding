package main

import "sort"

// 两数之和
func twoSum(nums []int, target int) []int {
	calcMap := make(map[int]int, 0)
	for i, n := range nums {
		if value, ok := calcMap[n]; ok {
			return []int{value, i}
		}

		calcMap[target-n] = i
	}

	return nil
}

//

// 最长连续序列
func longestConsecutive(nums []int) int {
	numsMap := map[int]bool{}
	for _, n := range nums {
		numsMap[n] = true
	}

	var longestLength int
	for n := range numsMap {
		if numsMap[n-1] {
			continue
		}

		currentNum := n
		currentLength := 1
		for numsMap[currentNum+1] {
			currentNum++
			currentLength++
		}

		if longestLength < currentLength {
			longestLength = currentLength
		}
	}

	return longestLength
}

// 字母异位词分组
func groupAnagrams(strs []string) [][]string {
	strMap := map[[26]int][]string{}
	for _, s := range strs {
		count := [26]int{}
		for _, i := range s {
			count[i-'a']++
		}

		strMap[count] = append(strMap[count], s)
	}

	answer := make([][]string, 0, 0)
	for _, v := range strMap {
		answer = append(answer, v)
	}

	return answer
}

// 移动零
func moveZeroes(nums []int) {
	i := 0
	j := 0
	for ; i < len(nums); i++ {
		if nums[i] == 0 {
			continue
		}

		nums[j] = nums[i]
		j++
	}

	for ; j < len(nums); j++ {
		nums[j] = 0
	}
}

// 盛水最多的容器
func maxArea(height []int) int {
	i := 0
	j := len(height) - 1
	minNum := func(x, y int) int {
		if y < x {
			return y
		}

		return x
	}

	maxSize := minNum(height[i], height[j]) * (j - i)
	for i < j {
		if height[i] < height[j] {
			i++
		} else {
			j--
		}

		currentSize := minNum(height[i], height[j]) * (j - i)
		if currentSize > maxSize {
			maxSize = currentSize
		}
	}

	return maxSize
}

// 三数之和
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

			for j < k && nums[i]+nums[j]+nums[k] > 0 {
				k--
			}

			if j == k {
				break
			}

			if nums[i]+nums[j]+nums[k] == 0 {
				answer = append(answer, []int{nums[i], nums[j], nums[k]})
			}
		}
	}

	return answer
}

func trap(height []int) int {
	l := len(height)
	if l == 0 {
		return 0
	}

	leftMaxWater := make([]int, l)
	rightMaxWater := make([]int, l)
	leftMaxWater[0] = height[0]
	rightMaxWater[l-1] = height[l-1]

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

	for i := 1; i < l; i++ {
		leftMaxWater[i] = maxNum(leftMaxWater[i-1], height[i])
	}

	for i := l - 2; i >= 0; i-- {
		rightMaxWater[i] = maxNum(rightMaxWater[i+1], height[i])
	}

	var answer int
	for i, n := range height {
		answer += minNum(leftMaxWater[i], rightMaxWater[i]) - n
	}

	return answer
}
