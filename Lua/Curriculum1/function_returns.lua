-- Function with multiple return values in Lua
function getMinMax(numbers)
    local min = numbers[1]
    local max = numbers[1]
    
    for i = 2, #numbers do
        if numbers[i] < min then
            min = numbers[i]
        end
        if numbers[i] > max then
            max = numbers[i]
        end
    end
    
    return min, max
end

local values = {7, 2, 9, 4, 5}
local minimum, maximum = getMinMax(values)

print("In the given set, the minimum is " .. minimum .. " and the maximum is " .. maximum)
