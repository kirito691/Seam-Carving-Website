$(function() {
	$(".div1 > .v1").draggable({
		containment: ".div1",
		axis: "x"
	});

	$(".div1 > .h1").draggable({
		containment: ".div1",
		axis: "y"
	})

	$("#resize-width-button").click(function(){
		var w = $(".v1").css("left");
		$.post('/width', {
			width = w
		}).done(function(response){
			$(destElem).html('<img src="{{ url_for('img', filename='scream.jpg') }}">');
		}).fail(function(){
			$(destElem).text("{{ _('Error: Could not contact server.') }}");
		});
	})

	$("#resize-height-button").click(function(){
		var h = $(".h1").css("top");
		$("#img").css("height", h);
	})
});
//var w = $(".v1").css("left");
		// $.post('/width', {
		// 	width = w
		// }).done(function(response){
		// 	$(destElem).text("{{ _('The deed has been done.') }}");
		// }).fail(function(){
		// 	$(destElem).text("{{ _('Error: Could not contact server.') }}");
		// });