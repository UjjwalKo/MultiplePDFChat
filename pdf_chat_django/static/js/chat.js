// Assuming jQuery is included in your HTML file.  If not, include it: <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>

$(document).ready(() => {
  $("#pdf-upload-form").submit(function (e) {
    e.preventDefault()
    var formData = new FormData(this)
    $("#upload-status").text("Processing...").removeClass("text-red-500 text-green-500").addClass("text-blue-500")
    $.ajax({
      url: "process_pdfs/",
      type: "POST",
      data: formData,
      processData: false,
      contentType: false,
      success: (response) => {
        if (response.status === "success") {
          $("#upload-status")
            .text("PDFs processed successfully!")
            .removeClass("text-red-500 text-blue-500")
            .addClass("text-green-500")
        } else {
          $("#upload-status")
            .text("Error: " + response.message)
            .removeClass("text-green-500 text-blue-500")
            .addClass("text-red-500")
        }
      },
      error: () => {
        $("#upload-status")
          .text("Error uploading PDFs.")
          .removeClass("text-green-500 text-blue-500")
          .addClass("text-red-500")
      },
    })
  })

  $("#chat-form").submit((e) => {
    e.preventDefault()
    var userQuestion = $("#user-question").val()
    if (userQuestion) {
      $("#chat-container").append('<p class="mb-2"><strong>You:</strong> ' + userQuestion + "</p>")
      $("#chat-container").append(
        '<p class="mb-2"><strong>AI:</strong> <span class="text-gray-500">Thinking...</span></p>',
      )
      $("#chat-container").scrollTop($("#chat-container")[0].scrollHeight)
      $.ajax({
        url: "/chat/",
        type: "POST",
        data: {
          question: userQuestion,
          csrfmiddlewaretoken: $("input[name=csrfmiddlewaretoken]").val(),
        },
        success: (response) => {
          if (response.status === "success") {
            $("#chat-container p:last").html("<strong>AI:</strong> " + response.response)
          } else {
            $("#chat-container p:last").html(
              '<strong>AI:</strong> <span class="text-red-500">Error: ' + response.message + "</span>",
            )
          }
          $("#chat-container").scrollTop($("#chat-container")[0].scrollHeight)
        },
        error: () => {
          $("#chat-container p:last").html(
            '<strong>AI:</strong> <span class="text-red-500">Error: Unable to get a response.</span>',
          )
          $("#chat-container").scrollTop($("#chat-container")[0].scrollHeight)
        },
      })
      $("#user-question").val("")
    }
  })
})