# RE603-Machine-Learning

Week 2 
Week 2 membahas dasar-dasar Python sebagai fondasi untuk Machine Learning.

## Materi

### 1. Tipe Data Python

Mempelajari tipe data dasar:

* int
* float
* string
* boolean
* list
* tuple
* dictionary
* set

Fokus pada pemahaman perbedaan tipe data serta konsep mutable dan immutable.

---

### 2. Function pada Python

Mempelajari:

* Cara membuat function dengan `def`
* Parameter dan argument
* Return value
* Pemanggilan function

Tujuan: membuat kode lebih terstruktur dan modular.

---

### 3. Object-Oriented Programming (OOP)

Mempelajari konsep dasar:

* Class
* Object
* Attribute
* Method
* Constructor (`__init__`)

#### Implementasi Encapsulation, Inheritance, dan Polymorphism

```python
class Vehicle:
    def __init__(self, brand, year):
        # Encapsulation
        self._brand = brand
        self._year = year

    def get_info(self):
        return f"{self._brand} - {self._year}"

    def start_engine(self):
        print("Engine started")


# Inheritance
class Car(Vehicle):
    # Polymorphism (method overriding)
    def start_engine(self):
        print("Car engine started with key")


class Motorcycle(Vehicle):
    # Polymorphism (method overriding)
    def start_engine(self):
        print("Motorcycle engine started with button")


if __name__ == "__main__":
    car = Car("Toyota", 2023)
    motor = Motorcycle("Honda", 2022)

    print(car.get_info())
    print(motor.get_info())

    car.start_engine()
    motor.start_engine()
```

---

## Tools

* Python 3
* Jupyter Notebook / VSCode

---

**Machine Learning RE603 — Week 2**
