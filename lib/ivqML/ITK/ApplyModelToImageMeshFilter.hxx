// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__ITK__ApplyModelToImageMeshFilter__hxx__
#define __ivqML__ITK__ApplyModelToImageMeshFilter__hxx__

// -------------------------------------------------------------------------
template< class _TInput, class _TModel >
const typename ivqML::ITK::ApplyModelToImageMeshFilter< _TInput, _TModel >::
TModel* ivqML::ITK::ApplyModelToImageMeshFilter< _TInput, _TModel >::
GetModel( ) const
{
  return( this->m_Model );
}

// -------------------------------------------------------------------------
template< class _TInput, class _TModel >
void ivqML::ITK::ApplyModelToImageMeshFilter< _TInput, _TModel >::
SetModel( const TModel& m )
{
  this->m_Model = &m;
  this->Modified( );
}

// -------------------------------------------------------------------------
template< class _TInput, class _TModel >
ivqML::ITK::ApplyModelToImageMeshFilter< _TInput, _TModel >::
ApplyModelToImageMeshFilter( )
  : Superclass( )
{
}

// -------------------------------------------------------------------------
template< class _TInput, class _TModel >
void ivqML::ITK::ApplyModelToImageMeshFilter< _TInput, _TModel >::
GenerateOutputInformation( )
{
  // Compute number of axes
  const TInput* i = this->GetInput( );

  // Configure output
  TOutput* o = this->GetOutput( );
  o->SetLargestPossibleRegion( i->GetLargestPossibleRegion( ) );
  o->SetRequestedRegion( i->GetRequestedRegion( ) );
  o->SetSpacing( i->GetSpacing( ) );
  o->SetOrigin( i->GetOrigin( ) );
  o->SetDirection( i->GetDirection( ) );
  o->SetNumberOfComponentsPerPixel( this->m_Model->number_of_outputs( ) );
}

// -------------------------------------------------------------------------
template< class _TInput, class _TModel >
void ivqML::ITK::ApplyModelToImageMeshFilter< _TInput, _TModel >::
GenerateData( )
{
  using _M = typename TModel::TMatrix;

  this->AllocateOutputs( );
  const TInput* input = this->GetInput( );
  TOutput* output = this->GetOutput( );

  _M I = ivqML::Common::ImageHelpers::meshgrid< TInput, _M >( input );
  auto O = ivq::ITK::ImageToMatrix( output ).transpose( );
  O = this->m_Model->evaluate( I );
}

#endif // __ivqML__ITK__ApplyModelToImageMeshFilter__hxx__

// eof - $RCSfile$
