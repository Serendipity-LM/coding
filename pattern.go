package main

/* 回溯模板，先确定终止条件，再确定横向遍历元素，纵向递归回溯元素，确定函数参数
func backTrack() {
	if "终止条件" {
		转化、存储结果
		如：copy(dst []int, src []int)
		ans = append(ans, dst)
		return
	}

	for (横向遍历：本层集合中元素（树中节点孩子的数量就是集合的大小）) {
		处理节点
		backTrack(路径，选择列表); // 递归，纵向遍历
		回溯，撤销处理结果
	}
}
*/
