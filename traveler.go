package main

import (
	"fmt"
	"github.com/fighterlyt/permutation"
)

type City struct {
	name string
	lat  int
	lon  int
}

func NewCity(n string, x int, y int) *City {
	return &City{name: n, lat: x, lon: y}
}

func ReadCities(path string) ([]*City, error) {
	/* TODO: Leer las ciudades de un archivo o algo asi */
	cities := []*City{NewCity("a", 15, 78), NewCity("b", 81, 92),
		NewCity("c", 65, 3), NewCity("d", 31, 34)}
	return cities, nil
}

func lessCity(i, j interface{}) bool {
	return true
}

func (c City) String() string {
	return fmt.Sprintf("%s: %d,%d", c.name, c.lat, c.lon)
}

func main() {
	cities, err := ReadCities("huehuehuehuehuehue")
	p, err := permutation.NewPerm(cities, lessCity) //generate a Permutator
	if err != nil {
		fmt.Println(err)
		return
	}
	for c, err := p.Next(); err == nil; c, err = p.Next() {
		//fmt.Printf("%3d permutation: %v left %d\n", p.Index()-1, i.([]int), p.Left())
		citiesPermutation := c.([]*City)
		for _, perm := range citiesPermutation {
			fmt.Println(perm)
		}
		fmt.Printf("permutation: %d left %d\n", p.Index()-1, p.Left())
	}
}
