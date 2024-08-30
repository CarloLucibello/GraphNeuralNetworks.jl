using DataFrames, CSV
dfall = CSV.read("contacts (4).csv", DataFrame)
df = dfall[!,["Last Name", "First Name"]]
# In column "Labels" replace everything after : with nothing
df[!,"Group"] = replace.(dfall[!,"Labels"], r" :.*" => "")
df[!,"Email"] = dfall[!, "E-mail 1 - Value"]
df[!,"Org"] = dfall[!, "Organization Name"]
# sort by last name
sort!(df, ["Last Name", "First Name"])

CSV.write("contacts.csv", df)