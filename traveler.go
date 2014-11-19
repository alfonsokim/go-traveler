package main

import (
	"bytes"
	"fmt"
	"github.com/fighterlyt/permutation"
	"math"
)

type City struct {
	name string
	lat  int
	lon  int
}
type Path struct {
	cities   []*City
	distance float64
}

func NewCity(n string, x int, y int) *City {
	return &City{name: n, lat: x, lon: y}
}

func NewPath(c []*City, d float64) *Path {
	return &Path{cities: c, distance: d}
}

func ReadCities(path string) ([]*City, error) {
	/* TODO: Leer las ciudades de un archivo o algo asi */
	cities := []*City{NewCity("a", 15, 78), NewCity("b", 81, 92),
		NewCity("c", 65, 3), NewCity("d", 31, 34)}
	return cities, nil
}

func lessCity(i, j interface{}) bool {
	return true // no importa el orden en el que devuelve las permutaciones
}

func (c City) String() string {
	return fmt.Sprintf("%s: %d,%d", c.name, c.lat, c.lon)
}

func (p Path) String() string {
	var buffer bytes.Buffer
	for i := 0; i < len(p.cities)-1; i++ {
		buffer.WriteString(p.cities[i].name)
		buffer.WriteString(" -> ")
	}
	buffer.WriteString(p.cities[len(p.cities)-1].name)
	buffer.WriteString(": ")
	buffer.WriteString(fmt.Sprintf("%.2f", p.distance))
	return buffer.String()
}

func (c City) Distance(other City) float64 {
	x := float64(c.lat - other.lat)
	y := float64(c.lon - other.lon)
	return math.Sqrt(x*x + y*y)
}

func main() {
	cities, err := ReadCities("huehuehuehuehuehue")
	p, err := permutation.NewPerm(cities, lessCity)
	if err != nil {
		fmt.Println(err)
		return
	}

	paths := make([]Path, p.Left())
	for c, err := p.Next(); err == nil; c, err = p.Next() {
		citiesPermutation := c.([]*City)
		totalDistance := float64(0)
		for i := 0; i < len(citiesPermutation)-1; i++ {
			from := citiesPermutation[i]
			to := citiesPermutation[i+1]
			distance := from.Distance(*to)
			//fmt.Println("de ", from, " a ", to, " = ", distance)
			totalDistance += distance
		}
		paths[p.Index()-1] = *NewPath(citiesPermutation, totalDistance)
		//fmt.Printf("permutation: %d left %d\n", p.Index()-1, p.Left())
	}
	for _, path := range paths {
		fmt.Println("Path: ", path)
	}
}
