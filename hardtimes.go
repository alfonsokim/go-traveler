package main

import (
	"bytes"
	"fmt"
	"github.com/fighterlyt/permutation"
	"math"
	"math/rand"
	"runtime"
	"time"
)

type City struct {
	name string
	lat  int
	lon  int
}
type Path struct {
	cities   []*City
	distance float64
	best     bool
	worst    bool
}

func NewCity(n string, x int, y int) *City {
	return &City{name: n, lat: x, lon: y}
}

func NewPath(c []*City, d float64) *Path {
	return &Path{cities: c, distance: d, best: false, worst: false}
}

func ReadCities(path string) ([]*City, error) {
	/* TODO: Leer las ciudades de un archivo o algo asi */
	numCities := 5
	cities := make([]*City, numCities)
	for i := 0; i < numCities; i++ {
		name := string(fmt.Sprintf("%d", i))
		lat := rand.Intn(90) + 5
		lon := rand.Intn(90) + 5
		cities[i] = NewCity(name, lat, lon)
	}
	return cities, nil
}

func lessCity(i, j interface{}) bool {
	city1 := i.(*City)
	city2 := j.(*City)
	return bytes.Compare([]byte(city1.name), []byte(city2.name)) < 0
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

func BuildPath(cities []*City) *Path {
	totalDistance := float64(0)
	for i := 0; i < len(cities)-1; i++ {
		from := cities[i]
		to := cities[i+1]
		distance := from.Distance(*to)
		totalDistance += distance
	}
	return NewPath(cities, totalDistance)
}

func ConcurrentBuildPath(paths chan *Path, cities []*City) {
	go func() {
		totalDistance := float64(0)
		for i := 0; i < len(cities)-1; i++ {
			from := cities[i]
			to := cities[i+1]
			distance := from.Distance(*to)
			totalDistance += distance
		}
		paths <- NewPath(cities, totalDistance)
	}()
}

func SequentialBuildPath(paths chan *Path, cities []*City) {
	totalDistance := float64(0)
	for i := 0; i < len(cities)-1; i++ {
		from := cities[i]
		to := cities[i+1]
		distance := from.Distance(*to)
		totalDistance += distance
	}
	paths <- NewPath(cities, totalDistance)
}

func ExplorePaths(paths <-chan *Path) *Path {
	var bestPath *Path
	bestDistance := math.Inf(1)
	for path := range paths {
		if path.distance < bestDistance {
			bestPath = path
			bestDistance = path.distance
		}
	}
	return bestPath
}

func GeneratePaths(cities []*City) (<-chan *Path, error) {
	p, err := permutation.NewPerm(cities, lessCity)
	if err != nil {
		fmt.Println("Error al generar permutaciones: ", err)
		return nil, err
	}
	out := make(chan *Path)
	go func() {
		for c, err := p.Next(); err == nil; c, err = p.Next() {
			citiesPerm := c.([]*City)
			out <- BuildPath(citiesPerm)
		}
		close(out)
	}()
	return out, nil
}

func ConcurrentGeneratePaths(cities []*City) (<-chan []*City, error) {
	p, err := permutation.NewPerm(cities, lessCity)
	if err != nil {
		fmt.Println("Error al generar permutaciones: ", err)
		return nil, err
	}
	out := make(chan []*City)
	go func() {
		for c, err := p.Next(); err == nil; c, err = p.Next() {
			//go func(out chan []*City, c interface{}) {
			out <- c.([]*City)
			//}(out, c)
		}
		close(out)
	}()
	return out, nil
}

func ConcurrentExplorePaths(paths <-chan []*City) <-chan *Path {
	out := make(chan *Path)
	go func() {
		for path := range paths {
			out <- BuildPath(path)
		}
		close(out)
	}()
	return out
}

func factorial(i int) int {
	result := 1
	for i > 0 {
		result *= i
		i--
	}
	return result
}

func SimplePathExplore(cities []*City) *Path {
	p, err := permutation.NewPerm(cities, lessCity)
	if err != nil {
		fmt.Println("Error al generar permutaciones: ", err)
		return nil
	}

	numPaths := factorial(len(cities))
	paths := make([]*Path, numPaths)

	sem := make(chan int, numPaths)

	i := 0
	for c, err := p.Next(); err == nil; c, err = p.Next() {
		cities := c.([]*City)
		go func(i int) {
			paths[i] = BuildPath(cities)
			sem <- 1
		}(i)
		i++
	}

	for i := 0; i < numPaths; i++ {
		<-sem
	}

	var bestPath *Path
	bestDistance := math.Inf(1)
	for _, path := range paths {
		if path.distance < bestDistance {
			bestPath = path
			bestDistance = path.distance
		}
	}
	return bestPath
}

func main() {
	runtime.GOMAXPROCS(1)
	//rand.Seed(time.Now().UnixNano())
	cities, err := ReadCities("huehuehuehuehuehue")
	if err != nil {
		fmt.Println("Error al leer ciudades: ", err)
		return
	}
	start := time.Now()
	/*
		citiesChan, _ := ConcurrentGeneratePaths(cities)
		pathsChan := ConcurrentExplorePaths(citiesChan)

		bestPath := ExplorePaths(pathsChan)
	*/

	bestPath := SimplePathExplore(cities)

	elapsed := time.Since(start)
	fmt.Println("Recorrer todos los camimos", elapsed)
	fmt.Println("Mejor camino:", bestPath)
}
