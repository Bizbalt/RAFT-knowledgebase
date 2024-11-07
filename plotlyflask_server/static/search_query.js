const url = window.location.origin + "/"


/**
 * Initializes the dropdown menus with options fetched from the server.
 *
 * This function sends a request to the server to get the dropdown options,
 * parses the response, and populates the dropdown menus for monomer, RAFT agent,
 * and solvent with the received options.
 *
 * @returns {Promise<void>} A promise that resolves when the dropdown menus are successfully populated.
 */
async function initializeDropdownOptions() {
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

/**
 * Initializes and reformats the results table.
 *
 * This function searches for the table in the results div, applies Bootstrap classes,
 * adds click event listeners to the table headers for sorting, and recolors rows with a score of 0.
 * It also ensures the table is displayed if found.
 *
 * @returns {Promise} A promise that resolves when the table is successfully reformatted or rejects if no table is loaded.
 */
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

// initialize the form dropdown options and the table reformatting with bootstrap.
// Also add the event listener for the search field and prohibit illegal checkbox choices
window.onload = function() {
    initializeDropdownOptions().then(() => console.log("Dropdowns initialized"))

    const input_field = document.getElementById("search");
    input_field.addEventListener("keyup", ({key}) => {
        if (key === "Enter") {
            plot_exp().then(() => console.log("Plotting"))
        }
    })
    table_reformat_init().then(() => console.log("Table class reformatted"), error => console.log(error))

    void prohibit_illegal_checkbox_choices()
}

/**
 * Sends a plot request to the server and updates the plot results.
 *
 * This function retrieves the experiment numbers from the search input field,
 * gathers the selected plot options (conversion, Mn, Mw, fit curves, stacked plots),
 * and sends a POST request to the server to generate the plot. It then displays
 * a loading GIF while waiting for the server response, and finally updates the plot
 * results using Plotly.
 */
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

/**
 * Prohibits illegal checkbox choices by enabling or disabling related checkboxes
 * based on the current selections. It also manages the display of additional options
 * for conversion and derivative fit checkboxes.
 */
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
