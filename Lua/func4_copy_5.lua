function greet(name)
    return "Hello, " .. (name or "stranger")
end
print(greet())  
print(greet("Lua")) 
