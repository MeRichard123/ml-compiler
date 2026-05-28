-- String manipulation functions in Lua
local message = "Hello, Lua Programming!"

-- String length
local length = string.len(message)
-- Uppercase conversion
local upper = string.upper(message)
-- Find substring position
local position = string.find(message, "Lua")

print("The word 'Lua' starts at position " .. position .. " in the string: " .. message)
