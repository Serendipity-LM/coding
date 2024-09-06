package main

// Stack 栈
/*
	interface转具体的类型
	value := reflect.ValueOf(n)
	if value.Kind() == reflect.Pointer {
		interfaceVal := value.Interface()
		if myStructVal, ok := interfaceVal.(*Node); ok {
			fmt.Println(myStructVal.Val)
		}
	}
*/
type Stack struct {
	Vals []interface{}
}

// 链表
type ListNode struct {
	Val  int
	Next *ListNode
}

// 随机链表
type Node struct {
	Val    int
	Next   *Node
	Random *Node
}

// 双向链表
type TwoWayListNode struct {
	Key   int
	Value int
	Next  *TwoWayListNode
	Pre   *TwoWayListNode
}

// 二叉树
type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
	Next  *TreeNode
}

func (s *Stack) Push(v interface{}) {
	s.Vals = append(s.Vals, v)
}

func (s *Stack) Pop() interface{} {
	if len(s.Vals) == 0 {
		return -1
	}

	v := s.Vals[len(s.Vals)-1]
	s.Vals = s.Vals[:len(s.Vals)-1]
	return v
}

func (s *Stack) IsEmpty() bool {
	return len(s.Vals) == 0
}
