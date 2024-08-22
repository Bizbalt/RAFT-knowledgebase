const url = window.location.origin + "/"


// building the dropdown menus
async function options_init() {
    const response = await fetch(url + 'get_dropdown_options')
    const dropdown_options_dic = await response.json()

    const monomerSel = document.getElementById("monomer");
    const raft_agentSel = document.getElementById("raft_agent");
    const solventSel = document.getElementById("solvent");

    const dropdownSelectors = {"monomer": monomerSel, "raft_agent": raft_agentSel, "solvent": solventSel}
    const keys = ["monomer", "raft_agent", "solvent"]

    for (const dropdown of keys){
        const dropdownSel = dropdownSelectors[dropdown]
        for (const option of dropdown_options_dic[dropdown]){
            dropdownSel.options[dropdownSel.options.length] = new Option(option, option);
        }
    }
}

async function table_reformat_init() {
    // search for the table in the results div
    const table = document.getElementById("results_div_table").getElementsByClassName("dataframe")[0];
    if (table === undefined) {
        return Promise.reject("No table loaded.")}
    table.className = "table table-striped table-bordered table-hover table-sm";

    // add onclick for every header
    for (const [index, th] of Array.from(table.getElementsByTagName("th")).entries()){
        th.setAttribute("onclick", `sort_table(${index})`)
    }
    document.getElementById("results_div_table").style.display = "block";

    // recolor rows that have a score of 0 red-ish
    let rows = table.rows;
    // iterate through every but the first line
    for (let i = 1; i < (rows.length - 1); i++) {
        const row_elements = rows[i].getElementsByTagName("TD")
        // get the score row
        const score = row_elements["score"];
        // decide to change the color depending on the score bein higher than 0
        if (!(score > 0)){
            for (let ele in row_elements){
                console.log(ele)
                // color each element
            }
        }

    }
}

window.onload = function() {
    options_init().then(() => console.log("Dropdowns initialized"))

    const input_field = document.getElementById("search");
    input_field.addEventListener("keyup", ({key}) => {
        if (key === "Enter") {
            plot_exp().then(() => console.log("Plotting"))
        }
    })
    table_reformat_init().then(() => console.log("Table class reformatted"), error => console.log(error))
}

// function to send a plot request to the server
async function plot_exp() {
    const results_div = document.getElementById("results_div_plot");
    while (results_div.firstChild) {
        results_div.removeChild(results_div.firstChild);
    }

    let experiment_list = document.getElementById("search").value;
    // create a list of the experiments numbers as strings
    let experiments_list = experiment_list.split(" ").map(x => String(Number(x)))
    let plot_mn = document.getElementById("mn").checked
    let plot_mw = document.getElementById("mw").checked
    let fit_curves = [document.getElementById("fit_curve").checked, document.getElementById("fit_derivat_curve").checked]
    const response = await fetch(url + `plot_exp`, {
        method: "POST",
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ exp_nr: experiments_list, plot_mn: plot_mn, plot_mw: plot_mw, fit_curves: fit_curves })
        })

    const loading_gif = document.createElement("img");
    loading_gif.src = "static/loading_i_86x50.gif";
    results_div.appendChild(loading_gif);

    const content = await response.json()
    loading_gif.remove();
    Plotly.newPlot("results_div_plot",content.data, content.layout);

    document.getElementById("results_div_plot").scrollIntoView()
}
