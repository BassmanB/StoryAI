$(function () {


  function sentenceCase(input, lowercaseBefore) {
    input = ( input === undefined || input === null ) ? '' : input;
    if (lowercaseBefore) { input = input.toLowerCase(); }
    return input.toString().replace( /(^|\. *)([a-z])/g, function(match, separator, char) {
        return separator + char.toUpperCase();
    });
  }

  $(".writer").click(function (){

      var writer = this.id;
      var text = $("#special").val();
      console.log(text);
      text = text.toLowerCase();
      console.log(text);

      $("#loader").attr("hidden", false);
       //<img id="anton" src="{% static 'images/Anton_Chekhov.jpg' %}" class="img-thumbnail writer">

      $.ajax({
      url: "author_response", 
    		type: 'POST',
    		data: {text: text, writer: writer},
    		success: function(result){
    			console.log("hehe")
      			console.log(result);
             $("#loader").attr("hidden", true);
      		},
      		error: function(result){
      			console.log("nie takie hehe wcale");
      			console.log(result.responseText);
            $("#loader").attr("hidden", true);
            var text = sentenceCase(result.responseText);
      			$("#response").text(text);
      		}
      })

  });
});