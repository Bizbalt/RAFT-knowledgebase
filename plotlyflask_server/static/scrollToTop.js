document.addEventListener("DOMContentLoaded", function() {
  // Get the button:
  const topButton = document.getElementById("totop");

  // When the user scrolls down from the top of the document, show the button
  window.onscroll = function() {scrollFunction()};

  function scrollFunction() {
    if (document.body.scrollTop > 500 || document.documentElement.scrollTop > 500) {
      topButton.style.animation = "fadeIn 0.3s";
      topButton.style.opacity = "1";
    } else {
      topButton.style.animation = "fadeOut 0.3s";
      topButton.style.opacity = "0";
    }
  }
});

// When the user clicks on the button, scroll to the top of the document
function topFunction() {
  document.body.scrollTop = 0; // For Safari
  document.documentElement.scrollTop = 0; // For Chrome, Firefox, IE and Opera
}