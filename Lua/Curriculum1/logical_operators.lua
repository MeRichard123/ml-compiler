-- Logical operators in Lua
local isRaining = true
local hasUmbrella = false
local willGetWet = isRaining and not hasUmbrella

print("Will I get wet? " .. tostring(willGetWet))  -- Output: Will I get wet? true
