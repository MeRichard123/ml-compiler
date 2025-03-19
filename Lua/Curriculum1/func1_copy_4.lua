function say_hello(name)
    print("Hello, " .. (name or "world") .. "!")
end
say_hello()
