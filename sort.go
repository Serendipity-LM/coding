package main

// BubbleSort 冒泡排序
func BubbleSort(array []int) {
	for i := 0; i < len(array); i++ {
		isSort := false
		for j := 0; j < len(array)-i-1; j++ {
			if array[j] > array[j+1] {
				array[j], array[j+1] = array[j+1], array[j]
				isSort = true
			}
		}

		if !isSort {
			return
		}
	}

	return
}

// SelectionSort 选择排序
func SelectionSort(arr []int) {
	var minIndex int
	for i := 0; i < len(arr)-1; i++ {
		minIndex = i
		for j := i + 1; j < len(arr); j++ {
			if arr[j] < arr[minIndex] {
				minIndex = j
			}
		}

		arr[i], arr[minIndex] = arr[minIndex], arr[i]
	}

	return
}

// InsertionSort 插入排序
func InsertionSort(arr []int) {
	var preIndex, nowNum int
	for i := 1; i < len(arr); i++ {
		preIndex = i - 1
		nowNum = arr[i]
		for preIndex >= 0 && nowNum < arr[preIndex] {
			arr[preIndex+1] = arr[preIndex]
			preIndex--
		}

		arr[preIndex+1] = nowNum
	}

	return
}

// ShellSort 希尔排序
func ShellSort(arr []int) {
	gep := 1
	for gep < len(arr)/3 {
		gep = gep*3 + 1
	}

	for ; gep > 0; gep = gep / 3 {
		for i := gep; i < len(arr); i++ {
			var nowNum = arr[i]
			j := i - gep
			for ; j >= 0 && arr[j] > nowNum; j -= gep {
				arr[j+gep] = arr[j]
			}

			arr[j+gep] = nowNum
		}
	}

	return
}

// 归并排序
func MergeSort(arr []int, left, right int) {
	if left >= right {
		return
	}

	MergeSort(arr, left, (left+right)/2)
	MergeSort(arr, (left+right)/2+1, right)
	mergeArr(arr, left, (left+right)/2, right)
	return
}

func mergeArr(arr []int, left, mid, right int) {
	l := mid - left + 1
	r := right - mid
	leftArr := make([]int, l)
	rightArr := make([]int, r)

	for i := 0; i < l; i++ {
		leftArr[i] = arr[left+i]
	}

	for i := 0; i < r; i++ {
		rightArr[i] = arr[mid+i+1]
	}

	i, j, k := 0, 0, left
	for i < l && j < r {
		if leftArr[i] < rightArr[j] {
			arr[k] = leftArr[i]
			i++
		} else {
			arr[k] = rightArr[j]
			j++
		}

		k++
	}

	for i < l {
		arr[k] = leftArr[i]
		i++
		k++
	}

	for j < r {
		arr[k] = rightArr[j]
		j++
		k++
	}

	return
}

// QuickSort 快速排序
func QuickSort(arr []int, left, right int) {
	if left >= right {
		return
	}

	par := partition(arr, left, right)
	QuickSort(arr, left, par-1)
	QuickSort(arr, par+1, right)
}

func partition(arr []int, left, right int) int {
	pivot := arr[right]
	i := left - 1
	for j := left; j < right; j++ {
		if arr[j] < pivot {
			i++
			arr[i], arr[j] = arr[j], arr[i]
		}
	}

	arr[i+1], arr[right] = arr[right], arr[i+1]
	return i + 1
}

// HeapSort 堆排序
func HeapSort(arr []int) {
	for i := len(arr)/2 - 1; i >= 0; i-- {
		heapify(arr, i, len(arr))
	}

	for j := len(arr) - 1; j > 0; j-- {
		arr[0], arr[j] = arr[j], arr[0]
		heapify(arr, 0, j)
	}
}

func heapify(arr []int, index, length int) {
	temp := arr[index]
	for i := 2*index + 1; i < length; i = 2*i + 1 {
		if i+1 < length && arr[i+1] > arr[i] {
			i++
		}

		// 大顶堆
		if arr[i] > temp {
			arr[index] = arr[i]
			index = i
		} else {
			break
		}
	}

	arr[index] = temp
}

func heapifySmall(arr []int, index, length int) {
	temp := arr[index]
	for i := 2*index + 1; i < length; i = 2*i + 1 {
		if i+1 < length && arr[i+1] < arr[i] {
			i++
		}

		if arr[i] < temp {
			arr[index] = arr[i]
			index = i
		} else {
			break
		}
	}

	arr[index] = temp
}

// CountingSort 计数排序
func CountingSort(arr []int) {
	if len(arr) <= 1 {
		return
	}

	maxVal := arr[0]
	for i := 1; i < len(arr); i++ {
		if arr[i] > maxVal {
			maxVal = arr[i]
		}
	}

	countArr := make([]int, maxVal+1)
	for i := 0; i < len(arr); i++ {
		countArr[arr[i]]++
	}

	for i, j := 0, 0; i <= maxVal; i++ {
		for countArr[i] > 0 {
			arr[j] = i
			j++
			countArr[i]--
		}
	}

	return
}

// BucketSort 桶排序
func BucketSort(arr []int) {
	minNum, maxNum := arr[0], arr[0]
	for i := 0; i < len(arr); i++ {
		if arr[i] < minNum {
			minNum = arr[i]
		}

		if arr[i] > maxNum {
			maxNum = arr[i]
		}
	}

	bucketCount := (maxNum-minNum)/len(arr) + 1
	buckets := make([][]int, bucketCount)
	for i := 0; i < len(arr); i++ {
		n := (arr[i] - minNum) / len(arr)
		buckets[n] = append(buckets[n], arr[i])
		k := len(buckets[n]) - 2
		for k >= 0 && buckets[n][k] > arr[i] {
			buckets[n][k+1] = buckets[n][k]
			k--
		}

		buckets[n][k+1] = arr[i]
	}

	index := 0
	for i := 0; i < len(buckets); i++ {
		for j := 0; j < len(buckets[i]); j++ {
			arr[index] = buckets[i][j]
			index++
		}
	}

	return
}

// RadixSort 基数排序
func RadixSort(arr []int) {
	maxValue := arr[0]
	for i := 0; i < len(arr); i++ {
		if arr[i] > maxValue {
			maxValue = arr[i]
		}
	}

	for exp := 1; maxValue/exp > 0; exp *= 10 {
		radixSort(arr, exp)
	}
}

func radixSort(arr []int, exp int) {
	count := make([]int, 10)
	output := make([]int, len(arr))

	for i := 0; i < len(arr); i++ {
		count[(arr[i]/exp)%10]++
	}

	for i := 1; i < 10; i++ {
		count[i] += count[i-1]
	}

	for i := len(arr) - 1; i >= 0; i-- {
		index := (arr[i] / exp) % 10
		output[count[index]-1] = arr[i]
		count[index]--
	}

	for i := 0; i < len(arr); i++ {
		arr[i] = output[i]
	}
}
