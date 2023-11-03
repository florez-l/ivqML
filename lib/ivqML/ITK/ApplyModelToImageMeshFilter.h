// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__ITK__ApplyModelToImageMeshFilter__h__
#define __ivqML__ITK__ApplyModelToImageMeshFilter__h__

#include <itkImageToImageFilter.h>
#include <itkVectorImage.h>

namespace ivqML
{
  namespace ITK
  {
    /**
     */
    template< class _TInput, class _TModel >
    class ApplyModelToImageMeshFilter
      : public ::itk::ImageToImageFilter< _TInput, itk::VectorImage< typename _TModel::TScalar, _TInput::ImageDimension > >
    {
    public:
      using TInput   = _TInput;
      using TModel   = _TModel;
      using TReal    = typename TModel::TScalar;
      using TPixel   = typename TInput::PixelType;
      using TTraits  = itk::NumericTraits< TPixel >;
      using TChannel = typename TTraits::ValueType;
      using TOutput  = itk::VectorImage< TReal, TInput::ImageDimension >;

      using Self         = ApplyModelToImageMeshFilter;
      using Superclass   = itk::ImageToImageFilter< TInput, TOutput >;
      using Pointer      = itk::SmartPointer< Self >;
      using ConstPointer = itk::SmartPointer< const Self >;

    public:
      itkNewMacro( Self );
      itkTypeMacro(
        ivqML::ITK::ApplyModelToImageMeshFilter, itk::ImageToImageFilter
        );

    public:
      const TModel* GetModel( ) const;
      void SetModel( const TModel& m );

    protected:
      ApplyModelToImageMeshFilter( );
      virtual ~ApplyModelToImageMeshFilter( ) override = default;

      virtual void GenerateOutputInformation( ) override;
      virtual void GenerateData( ) override;

    private:
      ApplyModelToImageMeshFilter( const Self& ) = delete;
      Self& operator=( const Self& ) = delete;

    protected:
      const TModel* m_Model { nullptr };
    };
  } // end namespace
} // end namespace

#ifndef ITK_MANUAL_INSTANTIATION
#  include <ivqML/ITK/ApplyModelToImageMeshFilter.hxx>
#endif // ITK_MANUAL_INSTANTIATION

#endif // __ivqML__ITK__ApplyModelToImageMeshFilter__h__

// eof - $RCSfile$
