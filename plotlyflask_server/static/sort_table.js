function sort_table_style(column_number) { // bootstrap preserving O(n^2) sort
  let table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
  table = document.getElementById("results_div_table").children[0];
  switching = true;
  // Set the sorting direction to ascending:
  dir = "asc";
  /* Make a loop that will continue until
  no switching has been done: */
  while (switching) {
    // Start by saying: no switching is done:
    switching = false;
    rows = table.rows;
    /* Loop through all table rows (except the
    first, which contains table headers): */
    for (i = 1; i < (rows.length - 1); i++) {
      // Start by saying there should be no switching:
      shouldSwitch = false;
      /* Get the two elements you want to compare,
      one from current row and one from the next: */
      x = rows[i].getElementsByTagName("TD")[column_number];
      y = rows[i + 1].getElementsByTagName("TD")[column_number];
      /* Check if the two rows should switch place,
      based on the direction, asc or desc: */
      if (dir == "asc") {
        if (x.innerHTML.toLowerCase() > y.innerHTML.toLowerCase()) {
          // If so, mark as a switch and break the loop:
          shouldSwitch = true;
          break;
        }
      } else if (dir == "desc") {
        if (x.innerHTML.toLowerCase() < y.innerHTML.toLowerCase()) {
          // If so, mark as a switch and break the loop:
          shouldSwitch = true;
          break;
        }
      }
    }
    if (shouldSwitch) {
      /* If a switch has been marked, make the switch
      and mark that a switch has been done: */
      rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
      switching = true;
      // Each time a switch is done, increase this count by 1:
      switchcount ++;
    } else {
      /* If no switching has been done AND the direction is "asc",
      set the direction to "desc" and run the while loop again. */
      if (switchcount == 0 && dir == "asc") {
        dir = "desc";
        switching = true;
      }
    }
  }
}
function sort_table_fast(column_number) { // fast O (n*log(n)) sort
  let table = document.getElementById("results_div_table").children[0];
  let rows = Array.from(table.rows).slice(1); // Convert rows to array and exclude the header row
  let dir = table.getAttribute("data-sort-dir") || "asc"; // Get current sorting direction or default to "asc"

  // Sort the rows array based on the content in the specified column
  rows.sort((a, b) => {
    let x = a.getElementsByTagName("TD")[column_number].innerHTML.toLowerCase();
    let y = b.getElementsByTagName("TD")[column_number].innerHTML.toLowerCase();

    if (x < y) return dir === "asc" ? -1 : 1;
    if (x > y) return dir === "asc" ? 1 : -1;
    return 0;
  });

  // Re-arrange rows in the table based on the sorted order, preserving formatting
  rows.forEach(row => table.appendChild(row));

  // Toggle direction for the next sort
  dir = dir === "asc" ? "desc" : "asc";
  table.setAttribute("data-sort-dir", dir);
}

function sort_table(column_number){ // this function is called per onclick event on headers
  // depending on the size of the table the slow but bootstrap preserving function will be used
    let table_size = document.getElementById("results_div_table").children[0].rows.length;
    if (table_size < 50){
        sort_table_style(column_number);
    }
    else{
        sort_table_fast(column_number);
    }
}
