-- Local vs global variables in Lua
globalVar = "I am global"  -- Global variable (no 'local' keyword)

function showScope()
    local localVar = "I am local"  -- Local variable
    print(localVar)  -- Can access local variable inside its scope
    print(globalVar)  -- Can access global variable from anywhere
end

showScope()
print(globalVar)  -- Can access global variable outside the function
-- print(localVar)  -- This would cause an error as localVar is not accessible here
