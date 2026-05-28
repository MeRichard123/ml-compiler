-- Nested tables in Lua
local student = {
    name = "Alice",
    age = 22,
    grades = {
        math = 95,
        science = 88,
        history = 92
    }
}

print(student.name .. " scored " .. student.grades.math .. " in math")
