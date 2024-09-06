package main

type RabinKarp struct {
	pattern string
	hash    uint64
	prime   uint64
	mod     uint64
}

// NewRabinKarp returns a new RabinKarp instance
func NewRabinKarp(pattern string) *RabinKarp {
	return &RabinKarp{
		pattern: pattern,
		hash:    0,
		prime:   101, // a prime number
		mod:     256, // 2^8
	}
}

// Hash calculates the hash value of the pattern
func (rk *RabinKarp) Hash() uint64 {
	hash := uint64(0)
	for _, c := range rk.pattern {
		hash = (hash*uint64(c) + uint64(c)) % rk.mod
	}
	return hash
}

// Search searches for the pattern in the text
func (rk *RabinKarp) Search(text string) int {
	hash := rk.Hash()
	textHash := uint64(0)
	for i := 0; i < len(text); i++ {
		textHash = (textHash*uint64(text[i]) + uint64(text[i])) % rk.mod
		if i >= len(rk.pattern)-1 {
			if textHash == hash {
				// check if the pattern matches the text
				match := true
				for j := 0; j < len(rk.pattern); j++ {
					if text[i-j] != rk.pattern[len(rk.pattern)-1-j] {
						match = false
						break
					}
				}
				if match {
					return i - len(rk.pattern) + 1
				}
			}
			// remove the first character from the text hash
			textHash = (textHash - uint64(text[i-len(rk.pattern)+1])*uint64(rk.prime)) % rk.mod
			if textHash < 0 {
				textHash += rk.mod
			}
		}
	}
	return -1
}
