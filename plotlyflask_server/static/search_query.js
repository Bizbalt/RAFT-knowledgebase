const url = window.location.origin + "/"


// building the dropdown menus
async function options_init() {
    const response = await fetch(url + 'get_dropdown_options')
    const dropdown_options_dic = await response.json()

    const monomerSel = document.getElementById("monomer");
    const raft_agentSel = document.getElementById("raft_agent");
    const solventSel = document.getElementById("solvent");

    const dropdownSelectors = {"monomer": monomerSel, "raft_agent": raft_agentSel, "solvent": solventSel} //ToDo: Lets show the first letter in uppercase
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
    let table, rows
    table = document.getElementById("results_div_table").getElementsByClassName("dataframe")[0];
    if (table === undefined) {
        return Promise.reject("No table loaded.")}
    table.className = "table table-striped table-bordered table-hover table-sm";

    // add onclick for every header
    for (const [index, th] of Array.from(table.getElementsByTagName("th")).entries()){
        th.setAttribute("onclick", `sort_table(${index})`)
    }
    document.getElementById("results_div_table").style.display = "block";

    // recolor rows that have a score of 0 red-ish
    rows = table.rows;
    // iterate through every but the first line
    const score_col_num =  rows[1].getElementsByTagName("TD").length -1
    for (let i = 1; i < (rows.length); i++) {
        const row_elements = rows[i].getElementsByTagName("TD")
        // get the score row
        const score = row_elements[score_col_num].innerText;
        // decide to change the color depending on the score being higher than 0
        if (!parseFloat(score) > 0){
            for (let ele of row_elements){
                // color each element
                ele.style.backgroundColor = "#ffc8be"
                ele.style.padding = "0 0.25rem"
            }
        }

    }
}
//What does the "function" below do? --> please explain in a comment
window.onload = function() {
    options_init().then(() => console.log("Dropdowns initialized"))

    const input_field = document.getElementById("search");
    input_field.addEventListener("keyup", ({key}) => {
        if (key === "Enter") {
            plot_exp().then(() => console.log("Plotting"))
        }
    })
    table_reformat_init().then(() => console.log("Table class reformatted"), error => console.log(error))

    prohibit_illegal_checkbox_choices()
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
    let plot_conv = document.getElementById("conv").checked
    let plot_mn = document.getElementById("mn").checked
    let plot_mw = document.getElementById("mw").checked
    let fit_curves = [document.getElementById("fit_curve").checked, document.getElementById("fit_derivative_curve").checked]
    let stacked_plots = document.getElementById("stacked_plots").checked
    const response = await fetch(url + `plot_exp`, {
        method: "POST",
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ exp_nr: experiments_list, plot_conv:plot_conv, plot_mn: plot_mn, plot_mw: plot_mw, fit_curves: fit_curves, stacked_plots: stacked_plots })
        })

    const loading_gif = document.createElement("img");
    loading_gif.src = "static/loading_i_86x50.gif";
    results_div.appendChild(loading_gif);

    const content = await response.json()
    loading_gif.remove();
    Plotly.newPlot("results_div_plot",content.data, content.layout);

    document.getElementById("results_div_plot").scrollIntoView()
}
//ToDo: add explanation for the function below
async function prohibit_illegal_checkbox_choices(){
    const conv_neg_fit_checkbox = document.getElementById("fit_curve")
    const conv_neg_der_fit_checkbox = document.getElementById("fit_derivative_curve")
    const conv_checkbox = document.getElementById("conv")
    const conv_options_div = document.getElementById("conv_options")
    console.log(conv_checkbox)
    console.log(conv_options_div)

    function enable_conv_options(){
        if (conv_checkbox.checked){
            conv_options_div.style.display = "block";
        }
        else {
            conv_options_div.style.display = "none";
        }
    }
    conv_checkbox.addEventListener("change", enable_conv_options)

    function enable_neg_der_fit_box() {
        if (conv_neg_fit_checkbox.checked){
            conv_neg_der_fit_checkbox.disabled = false;
        }
        else {
            conv_neg_der_fit_checkbox.checked = false;
            conv_neg_der_fit_checkbox.disabled = true;
        }
    }
    conv_neg_fit_checkbox.addEventListener("change", enable_neg_der_fit_box)

    const mn_checkbox = document.getElementById("mn")
    const mw_checkbox = document.getElementById("mw")
    const stacked_plots_checkbox = document.getElementById("stacked_plots")

    function enable_stacked_plots_box(){
        let checked = 0
        for (const checkbox of [conv_checkbox, mn_checkbox, mw_checkbox]){
            if (checkbox.checked){
                checked += 1
            }
        }
        if (checked > 1){
            stacked_plots_checkbox.disabled = false;
        }
        else {
            stacked_plots_checkbox.checked = false;
            stacked_plots_checkbox.disabled = true;
        }
    }
    for (const checkbox of [conv_checkbox, mn_checkbox, mw_checkbox]){
        checkbox.addEventListener("change", enable_stacked_plots_box)
    }
}
