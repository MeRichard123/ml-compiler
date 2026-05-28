-- Table operations in Lua
local colors = {"red", "green", "blue"}
table.insert(colors, "yellow")  -- Add an element to the end of the table

print("The table now has " .. #colors .. " colors, and the last one is " .. colors[#colors])
