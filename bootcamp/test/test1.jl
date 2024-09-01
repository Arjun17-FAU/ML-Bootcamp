
function add_numbers(a,b)
    return a+b
end


ans  = add_numbers(2,3)
println("Hi Julia and the sum of 2 and 3 is $ans") 


array = [1,2,3,4,5]

avg = sum(array)/length(array)

println("Average of the array is $avg")

function double(x)
    return 2*x
end

function double_inp()
    println("Give an input:")

    input = readline()

    input_num = parse(Float64,input)

    result = double(input_num)

    print("The double of the number you gave is $result")
end

double_inp()