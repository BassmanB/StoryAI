$(function () {

  $("#anton").click(function (){

      var text = $("#special").val();
      console.log(text);

      $.ajax({
      url: "author_response", 
    		type: 'POST',
    		data: {text: text},
    		success: function(result){
    			console.log("hehe")
      			console.log(result);
      		},
      		error: function(result){
      			console.log("nie takie hehe wcale");
      			console.log(result.responseText);
      			$("#response").text(result.responseText);
      		}
      })

  });
});