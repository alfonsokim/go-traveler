package main

import (
	"bytes"
	"fmt"
	"github.com/GaryBoone/GoStats/stats"
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
	numCities := 7
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

func CopyPath(path *Path) *Path {
	newCities := make([]*City, len(path.cities))
	return NewPath(newCities, path.distance)
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

	semaphore := make(chan int, numPaths) // semaforo para notificar que los procesos terminaron
	defer close(semaphore)

	i := 0
	for c, err := p.Next(); err == nil; c, err = p.Next() {
		cities := c.([]*City)
		go func(i int) {
			paths[i] = BuildPath(cities)
			semaphore <- 1 // Notificar al semaforo que esta rutina ya acabo
		}(i)
		i++
	}

	for i := 0; i < numPaths; i++ {
		<-semaphore // Esperar por todos los procesos
	}

	// Encontrar el mejor camino
	var bestPath *Path
	bestDistance := math.Inf(1)
	for _, path := range paths {
		if path.distance < bestDistance {
			bestPath = CopyPath(path)
			bestDistance = path.distance
			path.cities = nil
		}
		path = nil
	}
	paths = nil
	return bestPath
}

func main() {
	//rand.Seed(time.Now().UnixNano())
	cities, err := ReadCities("huehuehuehuehuehue")
	if err != nil {
		fmt.Println("Error al leer ciudades: ", err)
		return
	}

	goProcs := []int{8, 4, 2, 1}
	for _, goProc := range goProcs {
		var statsBuffer stats.Stats
		runtime.GOMAXPROCS(goProc)
		for i := 0; i < 1000; i++ {
			start := time.Now()
			SimplePathExplore(cities)
			elapsed := time.Since(start)
			statsBuffer.Update(elapsed.Seconds())
		}
		fmt.Println("goProc:", goProc, ",promedio:", statsBuffer.Mean(), ",desviacion:", statsBuffer.SampleStandardDeviation())
	}
}


