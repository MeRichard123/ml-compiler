function is_even(n)
    return { is_even = n % 2 == 0 }
end
result = is_even(4)
print(result.is_even)
